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
"""Family-specific model output classes."""

from __future__ import annotations

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ModelOutputWithPast,
)


class FamilyModelOutputWithPast(Qwen3_5ModelOutputWithPast):
    r"""Pass-through for the base ``Model`` output (no loss/logits)."""


class FamilyCausalLMOutputWithPast(Qwen3_5CausalLMOutputWithPast):
    r"""Pass-through for ``ForConditionalGeneration`` output (loss + logits)."""
