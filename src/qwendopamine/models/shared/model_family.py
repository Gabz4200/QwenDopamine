"""Shared model family base classes — re-export shim.

The concrete class definitions live in dedicated per-concern modules:

  - :mod:`.pretrained` — :class:`FamilyPreTrainedModel`
  - :mod:`.vision`     — :class:`FamilyVisionModel`
  - :mod:`.outputs`    — :class:`FamilyModelOutputWithPast`
  - :mod:`.text`       — :class:`FamilyTextModel`, :class:`FamilyTextForSequenceClassification`
  - :mod:`.model`      — :class:`FamilyModel`
  - :mod:`.heads`      — :class:`FamilyForCausalLM`, :class:`FamilyForTokenClassification`,
                          :class:`FamilyForConditionalGeneration`, :class:`FamilyForSequenceClassification`
"""

from qwendopamine.models.shared.heads import (
    FamilyForCausalLM,
    FamilyForConditionalGeneration,
    FamilyForSequenceClassification,
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
