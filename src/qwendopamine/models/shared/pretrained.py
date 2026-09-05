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
"""Family-specific ``PreTrainedModel`` base class."""

from __future__ import annotations

from typing import ClassVar

from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextPreTrainedModel


class FamilyPreTrainedModel(Qwen3NextPreTrainedModel):
    r"""Base for family-specific ``PreTrainedModel`` subclasses."""

    _no_split_modules: ClassVar[list[str]] = []
    _can_record_outputs: ClassVar[dict[str, type]] = {}

    def _init_weights(self, module: nn.Module) -> None:
        r"""_init_weights(self, module: nn.Module) -> None

        Apply the default pretrained initialization, then family-specific hooks.

        Args:
            module (nn.Module): Module to initialize.
        """
        PreTrainedModel._init_weights(self, module)
        self._init_family_weights(module)

    def _init_family_weights(self, module: nn.Module) -> None:
        r"""_init_family_weights(self, module: nn.Module) -> None

        Override in subclasses to add family-specific weight initialization.

        Args:
            self - .
            module (nn.Module) - .
        """
