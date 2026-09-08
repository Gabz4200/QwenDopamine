"""Sequence classification head for Qwen-family models."""

from __future__ import annotations

from typing import Any

from transformers.modeling_layers import GenericForSequenceClassification

from qwendopamine.models.shared.pretrained import FamilyPreTrainedModel


class FamilyForSequenceClassification(
    GenericForSequenceClassification, FamilyPreTrainedModel
):
    r"""Base for family-specific ``ForSequenceClassification`` subclasses."""

    config_class: type
    config: Any
