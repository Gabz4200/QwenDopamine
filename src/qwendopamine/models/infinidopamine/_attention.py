"""InfiniDopamineAttention: full-attention layer with sliding-window support.

Moved from ``decoder_layer.py`` for size.
"""

from __future__ import annotations

from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextAttention

from qwendopamine.models.infinidopamine.configs import InfiniDopamineTextConfig


class InfiniDopamineAttention(Qwen3NextAttention):
    r"""InfiniDopamineAttention(config, layer_idx) -> None

    Standard full-attention layer with sliding-window support.

    Args:
        config (InfiniDopamineTextConfig): Layer configuration.
        layer_idx (int): Layer index.
    """

    def __init__(self, config: InfiniDopamineTextConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.sliding_window = getattr(config, "sliding_window", 1024)
        self.attention_dropout = getattr(
            config, "attention_dropout", getattr(config, "attention_dropout_prob", 0.0)
        )
