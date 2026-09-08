"""Shared base classes for Qwen-style model families."""

from qwendopamine.models.shared.heads_causal_lm import FamilyForCausalLM
from qwendopamine.models.shared.heads_conditional_generation import (
    FamilyForConditionalGeneration,
)
from qwendopamine.models.shared.heads_sequence_classification import (
    FamilyForSequenceClassification,
)
from qwendopamine.models.shared.heads_token_classification import (
    FamilyForTokenClassification,
)
from qwendopamine.models.shared.model import FamilyModel
from qwendopamine.models.shared.outputs import FamilyModelOutputWithPast
from qwendopamine.models.shared.pretrained import FamilyPreTrainedModel
from qwendopamine.models.shared.text import (
    FamilyTextForSequenceClassification,
    FamilyTextModel,
)
from qwendopamine.models.shared.vision import FamilyVisionModel

__all__ = [
    "FamilyForCausalLM",
    "FamilyForConditionalGeneration",
    "FamilyForSequenceClassification",
    "FamilyForTokenClassification",
    "FamilyModel",
    "FamilyModelOutputWithPast",
    "FamilyPreTrainedModel",
    "FamilyTextForSequenceClassification",
    "FamilyTextModel",
    "FamilyVisionModel",
]
