"""Token classification head for Qwen-family models."""

from __future__ import annotations

from typing import Any

from transformers.modeling_layers import GenericForTokenClassification

from qwendopamine.models.shared.pretrained import FamilyPreTrainedModel


class FamilyForTokenClassification(
    GenericForTokenClassification, FamilyPreTrainedModel
):
    r"""Base for family-specific ``ForTokenClassification`` subclasses."""

    config_class: type
    config: Any