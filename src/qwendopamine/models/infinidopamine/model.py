# Copyright 2025 The Qwen Team, InfiniDopamine Authors, and The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch InfiniDopamine model."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from transformers import initialization as init
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from qwendopamine.models._transformers_utils import unwrap_gated_delta_rule_fns
from qwendopamine.models.infinidopamine.configs import (
    InfiniDopamineConfig,
    InfiniDopamineTextConfig,
    InfiniDopamineVisionConfig,
)
from qwendopamine.models.infinidopamine.decoder_layer import (
    InfiniDopamineAttention,
    InfiniDopamineDecoderLayer,
    InfiniDopamineGatedDeltaNet,
    InfiniDopamineGatedRewardNet,  # noqa: F401 — re-exported for package API
)
from qwendopamine.models.infinidopamine.rotary_embeddings import (
    InfiniDopamineTextRotaryEmbedding,
    InfiniDopamineVisionRotaryEmbedding,
)
from qwendopamine.models.shared.model_family import (
    FamilyForCausalLM,
    FamilyForConditionalGeneration,
    FamilyForSequenceClassification,
    FamilyForTokenClassification,
    FamilyModel,
    FamilyModelOutputWithPast,
    FamilyPreTrainedModel,
    FamilyTextForSequenceClassification,
    FamilyTextModel,
    FamilyVisionModel,
)

unwrap_gated_delta_rule_fns()


def _normalize_weights_to_state_dict(
    weights: dict[str, torch.Tensor] | nn.Module,
) -> dict[str, torch.Tensor]:
    r"""Normalize ``weights`` into a plain ``state_dict``."""
    if isinstance(weights, nn.Module):
        return weights.state_dict()
    raw = dict(weights)
    if not all(isinstance(k, str) for k in raw):
        raise TypeError(
            f"weight dict keys must be str; got {[type(k).__name__ for k in raw if not isinstance(k, str)][:1]}"
        )
    return raw


class InfiniDopaminePreTrainedModel(FamilyPreTrainedModel):
    r"""InfiniDopaminePreTrainedModel: base pretrained model with GDN-2 weight
    initialization.

    Args:
        config (InfiniDopamineConfig): Model configuration.
    """

    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig
    _no_split_modules: ClassVar[list[str]] = [
        "InfiniDopamineDecoderLayer",
        "InfiniDopamineVisionBlock",
    ]
    _can_record_outputs: ClassVar[dict[str, type]] = {
        "hidden_states": InfiniDopamineDecoderLayer,
        "attentions": InfiniDopamineAttention,
    }

    def _init_family_weights(self, module: nn.Module) -> None:
        """Initialize GDN-2 and vision-rotary parameters."""
        if isinstance(module, InfiniDopamineGatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(
                module.A_log,
                torch.empty(module.num_v_heads, device=module.A_log.device)
                .uniform_(0.01, 16)
                .log_(),
            )
            init.zeros_(module.betas)
            init.zeros_(module.in_proj_gate.weight)
        elif isinstance(module, InfiniDopamineVisionRotaryEmbedding):
            inv_freq = 1.0 / (
                module.theta
                ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim)
            )
            init.copy_(module.inv_freq, inv_freq)


class InfiniDopamineVisionModel(FamilyVisionModel):
    r"""InfiniDopamineVisionModel: vision encoder for multimodal configs.

    Args:
        config (InfiniDopamineVisionConfig): Vision configuration.
    """

    config_class = InfiniDopamineVisionConfig
    config: InfiniDopamineVisionConfig
    _no_split_modules: ClassVar[list[str]] = ["InfiniDopamineVisionBlock"]

    def _delete_vision_attributes(self) -> None:
        """Remove vision-specific attributes for language-only inference."""
        del self.deepstack_visual_indexes
        del self.deepstack_merger_list


class InfiniDopamineModelOutputWithPast(FamilyModelOutputWithPast):
    r"""InfiniDopamineModelOutputWithPast: output container with past-key states."""


class InfiniDopamineTextModel(FamilyTextModel):
    r"""InfiniDopamineTextModel: text-only decoder stack with GDN-2 layers.

    Args:
        config (InfiniDopamineTextConfig): Text configuration.
    """

    config_class = InfiniDopamineTextConfig
    config: InfiniDopamineTextConfig

    def _build_text_layers(self, config: InfiniDopamineTextConfig) -> None:
        """Build the decoder layer stack and rotary embedding."""
        self.layers = nn.ModuleList(
            [
                InfiniDopamineDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rotary_emb = InfiniDopamineTextRotaryEmbedding(config=config)

    def _build_text_output(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache | None,
    ) -> BaseModelOutputWithPast:
        """Wrap hidden states and cache in an output container."""
        return InfiniDopamineModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def get_gate_regularization_loss(self, target: float = 0.5) -> torch.Tensor:
        r"""get_gate_regularization_loss(target=0.5) -> torch.Tensor

        Sum gate balance regularization across all GDN-2 layer blocks.

        Args:
            target (float): Target gate balance value. Default: ``0.5``.

        Returns:
            torch.Tensor: Scalar regularization loss.
        """
        losses: list[torch.Tensor] = []
        for layer in self.layers[: self.config.num_hidden_layers]:
            if hasattr(layer, "linear_attn") and hasattr(
                layer.linear_attn, "get_gate_regularization_loss"
            ):
                linear_attn: Any = layer.linear_attn
                losses.append(linear_attn.get_gate_regularization_loss(target=target))
        if not losses:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def load_qwen35_weights(
        self,
        weights: dict[str, torch.Tensor] | nn.Module,
        strict: bool = True,
    ) -> Any:
        r"""load_qwen35_weights(weights, strict=True) -> Any

        Load pretrained Qwen3.5 (GDN-1) weights into InfiniDopamine (GDN-2
        with SWA), stripping visual and MTP prefixes.

        Args:
            weights (dict[str, torch.Tensor] | nn.Module): State dict or module.
            strict (bool): Strict load. Default: ``True``.

        Returns:
            Any: Result of :meth:`load_state_dict` (missing/unexpected keys).
        """
        state_dict = _normalize_weights_to_state_dict(weights)

        has_full_prefix = any(
            k.startswith(("model.language_model.", "model.", "language_model."))
            for k in state_dict
        )
        if has_full_prefix:
            remapped_state_dict: dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                new_k = k
                if new_k.startswith("model.language_model."):
                    new_k = new_k[len("model.language_model.") :]
                elif new_k.startswith("language_model."):
                    new_k = new_k[len("language_model.") :]
                elif new_k.startswith("model."):
                    new_k = new_k[len("model.") :]
                if not (
                    k.startswith(("model.visual.", "visual.", "mtp."))
                    or k == "lm_head.weight"
                ):
                    remapped_state_dict[new_k] = v
            state_dict = remapped_state_dict

        return self.load_state_dict(state_dict, strict=strict)


class InfiniDopamineModel(FamilyModel):
    r"""InfiniDopamineModel: multimodal model combining vision and text towers.

    Args:
        config (InfiniDopamineConfig): Multimodal configuration.
    """

    config_class = InfiniDopamineConfig
    _no_split_modules: ClassVar[list[str]] = [
        "InfiniDopamineDecoderLayer",
        "InfiniDopamineVisionBlock",
    ]

    def _build_model_components(self, config: InfiniDopamineConfig) -> None:
        """Build vision and text submodels."""
        self.visual = InfiniDopamineVisionModel(config.vision_config)
        self.language_model = InfiniDopamineTextModel(config.text_config)


class InfiniDopamineForCausalLM(FamilyForCausalLM):
    r"""InfiniDopamineForCausalLM: causal LM head with optional gate/parallel-loss.

    Args:
        config (InfiniDopamineTextConfig): Text configuration.
    """

    config_class = InfiniDopamineTextConfig
    config: InfiniDopamineTextConfig
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
        r"^model.visual.*",
    ]

    def _build_causal_lm_model(self, config: InfiniDopamineTextConfig) -> nn.Module:
        """Return a :class:`InfiniDopamineTextModel`."""
        return InfiniDopamineTextModel(config)

    def _accumulate_aux_losses(self, loss: torch.Tensor | None) -> torch.Tensor | None:
        """Fold all enabled auxiliary losses (gate reg, parallel reward reg) into ``loss``.

        Single source of truth used by both the causal and the
        conditional post-processing paths. Returns the new total loss
        (``loss`` unchanged when no aux losses are enabled, the model
        is in eval mode, or no labelled loss was produced).
        """
        if loss is None or not self.training:
            return loss
        gate_weight = getattr(self.config, "gate_loss_weight", 0.0)
        if gate_weight > 0.0:
            target = getattr(self.config, "gate_target_balance", 0.5)
            loss = loss + gate_weight * self.get_gate_regularization_loss(target=target)
        if getattr(self.config, "use_parallel_reward", False):
            pr_weight = getattr(self.config, "parallel_reward_gate_loss_weight", 0.0)
            if pr_weight > 0.0:
                loss = loss + pr_weight * self.get_parallel_reward_gate_loss()
        return loss

    def _apply_causal_lm_postprocessing(self, outputs: CausalLMOutputWithPast) -> None:
        """Add gate regularization and parallel-reward losses to ``outputs.loss``."""
        outputs.loss = self._accumulate_aux_losses(getattr(outputs, "loss", None))

    def get_parallel_reward_gate_loss(self) -> torch.Tensor:
        r"""get_parallel_reward_gate_loss() -> torch.Tensor

        Mean ``σ(W_g x + b_g) - init_bias`` across all parallel reward gates.

        Penalises the gate from drifting away from its initialisation
        (``sigmoid(init_bias) ≈ 0.0067`` by default) so the dopamine branch
        stays effectively silent until the rest of the model has stabilised.
        Active layers only.

        Args:
            None

        Returns:
            torch.Tensor: Scalar penalty on gate deviation.
        """
        device = next(self.parameters()).device
        init_bias = float(getattr(self.config, "reward_gate_init_bias", -5.0))
        init_gate = float(torch.sigmoid(torch.tensor(init_bias)).item())
        # Use a fixed zero-like input so the gate ≈ sigmoid(bias) on init.
        # The exact value doesn't matter — only the deviation matters.
        losses: list[torch.Tensor] = []
        for layer in self.model.layers[: self.config.num_hidden_layers]:
            if not hasattr(layer, "reward_gate_proj"):
                continue
            gate = torch.sigmoid(layer.reward_gate_proj.bias)
            losses.append(((gate - init_gate) ** 2).mean())
        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def load_qwen35_weights(
        self,
        weights: dict[str, torch.Tensor] | nn.Module,
        strict: bool = True,
    ) -> Any:
        r"""load_qwen35_weights(weights, strict=True) -> Any

        Load Qwen3.5 (GDN-1) weights, remapping language-model prefixes.

        Args:
            weights (dict[str, torch.Tensor] | nn.Module): State dict or module.
            strict (bool): Strict load. Default: ``True``.

        Returns:
            Any: Result of :meth:`load_state_dict`.
        """
        state_dict = _normalize_weights_to_state_dict(weights)

        has_language_model_prefix = any(
            k.startswith(("model.language_model.", "language_model."))
            for k in state_dict
        )
        if has_language_model_prefix:
            remapped_state_dict: dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                if k.startswith("model.language_model."):
                    remapped_state_dict[
                        k.replace("model.language_model.", "model.")
                    ] = v
                elif k.startswith("language_model."):
                    remapped_state_dict[k.replace("language_model.", "model.")] = v
                elif k == "lm_head.weight" or not k.startswith(
                    ("model.visual.", "visual.", "mtp.")
                ):
                    remapped_state_dict[k] = v
            state_dict = remapped_state_dict

        return self.load_state_dict(state_dict, strict=strict)


class InfiniDopamineForTokenClassification(FamilyForTokenClassification):
    r"""InfiniDopamineForTokenClassification: token-classification head.

    Args:
        config (InfiniDopamineConfig): Model configuration.
    """

    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig


class InfiniDopamineForConditionalGeneration(FamilyForConditionalGeneration):
    r"""InfiniDopamineForConditionalGeneration: multimodal causal-LM head.

    Args:
        config (InfiniDopamineConfig): Multimodal configuration.
    """

    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig
    _keys_to_ignore_on_load_unexpected: ClassVar[list[str]] = [
        r"^mtp.*",
    ]

    def _build_conditional_model(self, config: InfiniDopamineConfig) -> nn.Module:
        """Return a :class:`InfiniDopamineModel`."""
        return InfiniDopamineModel(config)

    def _apply_conditional_postprocessing(
        self,
        loss: torch.Tensor | None,
        outputs: Any,
    ) -> None:
        """Add gate regularization loss to ``outputs.loss``.

        The conditional path delegates ``get_gate_regularization_loss``
        to the inner text tower (which has access to the per-layer
        state). The parallel-reward loss is not applied here because
        the conditional path is not yet wired to the per-layer parallel
        branch.
        """
        if loss is not None and self.training:
            weight = getattr(self.config, "gate_loss_weight", 0.0)
            if weight > 0.0:
                target = getattr(self.config, "gate_target_balance", 0.5)
                gate_loss = self.model.get_gate_regularization_loss(target=target)
                outputs.loss = loss + weight * gate_loss

    def get_gate_regularization_loss(self, target: float = 0.5) -> torch.Tensor:
        r"""get_gate_regularization_loss(target=0.5) -> torch.Tensor

        Delegate gate balance regularization to the model's text tower.

        Args:
            target (float): Target gate balance value. Default: ``0.5``.

        Returns:
            torch.Tensor: Scalar regularization loss.
        """
        result: torch.Tensor = self.model.get_gate_regularization_loss(target=target)
        return result

    def load_qwen35_weights(
        self,
        weights: dict[str, torch.Tensor] | nn.Module,
        strict: bool = True,
    ) -> _IncompatibleKeys:
        r"""load_qwen35_weights(weights, strict=True) -> _IncompatibleKeys

        Load Qwen3.5 weights into a multimodal model, splitting into vision,
        text, and LM-head state.

        Args:
            weights (dict[str, torch.Tensor] | nn.Module): State dict or module.
            strict (bool): Strict load. Default: ``True``.

        Returns:
            _IncompatibleKeys: Named tuple of ``missing_keys`` and
            ``unexpected_keys``.
        """
        state_dict = _normalize_weights_to_state_dict(weights)

        vision_state: dict[str, torch.Tensor] = {}
        text_state: dict[str, torch.Tensor] = {}
        lm_head_state: dict[str, torch.Tensor] = {}

        for k, v in state_dict.items():
            if k == "lm_head.weight":
                lm_head_state[k] = v
            elif k.startswith("model.visual."):
                vision_state[k[len("model.visual.") :]] = v
            elif k.startswith("model.language_model."):
                text_state[k[len("model.") :]] = v
            elif k.startswith("language_model."):
                text_state[k] = v
            elif k.startswith("visual."):
                vision_state[k[len("visual.") :]] = v
            elif k.startswith("mtp."):
                continue
            elif strict:
                text_state[k] = v

        load_info: list[str] = []
        all_missing: list[str] = []
        all_unexpected: list[str] = []

        if vision_state:
            missing_v, unexpected_v = self.model.visual.load_state_dict(
                vision_state, strict=strict
            )
            all_missing.extend(missing_v)
            all_unexpected.extend(unexpected_v)
            load_info.append(
                f"vision: loaded {len(vision_state) - len(missing_v)} keys "
                f"({len(missing_v)} missing, {len(unexpected_v)} unexpected)"
            )

        if text_state:
            missing_t, unexpected_t = self.model.language_model.load_qwen35_weights(
                text_state, strict=strict
            )
            all_missing.extend(missing_t)
            all_unexpected.extend(unexpected_t)
            load_info.append(
                f"text: loaded {len(text_state) - len(missing_t)} keys "
                f"({len(missing_t)} missing, {len(unexpected_t)} unexpected)"
            )

        if lm_head_state:
            self.lm_head.weight.data.copy_(lm_head_state["lm_head.weight"])
            load_info.append("lm_head: loaded 1 key")

        return _IncompatibleKeys(all_missing, all_unexpected)


class InfiniDopamineTextForSequenceClassification(FamilyTextForSequenceClassification):
    r"""InfiniDopamineTextForSequenceClassification: sequence classification on
    text input.

    Args:
        config (InfiniDopamineTextConfig): Text configuration.
    """

    config_class = InfiniDopamineTextConfig
    config: InfiniDopamineTextConfig
    input_modalities = ("text",)


class InfiniDopamineForSequenceClassification(FamilyForSequenceClassification):
    r"""InfiniDopamineForSequenceClassification: sequence classification on
    multimodal input.

    Args:
        config (InfiniDopamineConfig): Multimodal configuration.
    """

    config_class = InfiniDopamineConfig
    config: InfiniDopamineConfig


__all__ = [
    "InfiniDopamineConfig",
    "InfiniDopamineDecoderLayer",
    "InfiniDopamineForCausalLM",
    "InfiniDopamineForConditionalGeneration",
    "InfiniDopamineForSequenceClassification",
    "InfiniDopamineForTokenClassification",
    "InfiniDopamineGatedDeltaNet",
    "InfiniDopamineModel",
    "InfiniDopaminePreTrainedModel",
    "InfiniDopamineTextConfig",
    "InfiniDopamineTextForSequenceClassification",
    "InfiniDopamineTextModel",
    "InfiniDopamineTextRotaryEmbedding",
    "InfiniDopamineVisionConfig",
    "InfiniDopamineVisionModel",
    "InfiniDopamineVisionRotaryEmbedding",
]
