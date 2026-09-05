"""InfiniDopamineMLP: MLP block with hidden-state dropout.

Moved from ``decoder_layer.py`` for size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextMLP,
)

from qwendopamine.models.infinidopamine.configs import (
    InfiniDopamineConfig,
)


class InfiniDopamineMLP(Qwen3NextMLP):
    r"""InfiniDopamineMLP(config, intermediate_size) -> None

    MLP block with hidden-state dropout.

    Args:
        config (InfiniDopamineConfig): Layer configuration.
        intermediate_size (int): Feed-forward hidden dimension.
    """

    def __init__(self, config: InfiniDopamineConfig, intermediate_size: int) -> None:
        super().__init__(config, intermediate_size)
        self.intermediate_size = intermediate_size
        self.hidden_dropout = getattr(
            config, "hidden_dropout", getattr(config, "hidden_dropout_prob", 0.0)
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        r"""forward(hidden_state: torch.Tensor) -> torch.Tensor

        Apply gated MLP with optional training dropout.

        Args:
            hidden_state (torch.Tensor): Input ``[..., D]``.

        Returns:
            torch.Tensor: ``[..., D]`` output.
        """
        gate = self.act_fn(self.gate_proj(hidden_state))
        if self.training and self.hidden_dropout > 0.0:
            gate = F.dropout(gate, p=self.hidden_dropout, training=True)
        up = self.up_proj(hidden_state)
        down = self.down_proj(gate * up)
        if self.training and self.hidden_dropout > 0.0:
            down = F.dropout(down, p=self.hidden_dropout, training=True)
        result: torch.Tensor = down
        return result
