"""InfiniDopamine modular architecture."""

from qwendopamine.models.infinidopamine._attention import InfiniDopamineAttention
from qwendopamine.models.infinidopamine._gated_delta_net import (
    InfiniDopamineGatedDeltaNet,
)
from qwendopamine.models.infinidopamine._gated_reward_net import (
    InfiniDopamineGatedRewardNet,
)
from qwendopamine.models.infinidopamine._mlp import InfiniDopamineMLP
from qwendopamine.models.infinidopamine._norm import InfiniDopamineRMSNorm
from qwendopamine.models.infinidopamine.configs import (
    InfiniDopamineConfig,
    InfiniDopamineTextConfig,
    InfiniDopamineVisionConfig,
)
from qwendopamine.models.infinidopamine.decoder_layer import InfiniDopamineDecoderLayer
from qwendopamine.models.infinidopamine.model import (
    InfiniDopamineForCausalLM,
    InfiniDopamineForConditionalGeneration,
    InfiniDopamineForSequenceClassification,
    InfiniDopamineForTokenClassification,
    InfiniDopamineModel,
    InfiniDopaminePreTrainedModel,
    InfiniDopamineTextForSequenceClassification,
    InfiniDopamineTextModel,
    InfiniDopamineVisionModel,
)
from qwendopamine.models.infinidopamine.rotary_embeddings import (
    InfiniDopamineTextRotaryEmbedding,
    InfiniDopamineVisionRotaryEmbedding,
)

__all__ = [
    "InfiniDopamineAttention",
    "InfiniDopamineConfig",
    "InfiniDopamineDecoderLayer",
    "InfiniDopamineForCausalLM",
    "InfiniDopamineForConditionalGeneration",
    "InfiniDopamineForSequenceClassification",
    "InfiniDopamineForTokenClassification",
    "InfiniDopamineGatedDeltaNet",
    "InfiniDopamineGatedRewardNet",
    "InfiniDopamineMLP",
    "InfiniDopamineModel",
    "InfiniDopaminePreTrainedModel",
    "InfiniDopamineRMSNorm",
    "InfiniDopamineTextConfig",
    "InfiniDopamineTextForSequenceClassification",
    "InfiniDopamineTextModel",
    "InfiniDopamineTextRotaryEmbedding",
    "InfiniDopamineVisionConfig",
    "InfiniDopamineVisionModel",
    "InfiniDopamineVisionRotaryEmbedding",
]
