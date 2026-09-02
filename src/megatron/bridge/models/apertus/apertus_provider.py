# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Apertus Model Provider for Swiss AI Apertus models.

Runs on stock megatron-core: XIELU is owned by the bridge
(megatron.bridge.models.apertus.xielu_activation) and instantiated as a
module activation through the layer spec. Llama3-style RoPE scaling uses
the provider's native rope_scaling/rope_scaling_factor passthrough (mcore
hardcodes low_freq_factor=1.0, high_freq_factor=4.0, original context 8192
— the bridge validates the HF config matches those assumptions).
"""

import functools
import logging
from dataclasses import dataclass
from typing import Callable, Union

from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.apertus.xielu_activation import XIELU
from megatron.bridge.models.gpt_provider import GPTModelProvider, default_layer_spec


logger = logging.getLogger(__name__)


def apertus_layer_spec(config: "ApertusModelProvider") -> ModuleSpec:
    """Default layer spec with XIELU wired in as a module activation.

    The spec path (use_te_activation_func + MLPSubmodules.activation_func)
    makes mcore's MLP instantiate XIELU as a submodule, so its learnable
    alpha_p/alpha_n become ``mlp.activation_func.*`` parameters.
    """
    spec = default_layer_spec(config)
    mlp = spec.submodules.mlp
    if isinstance(mlp, functools.partial):
        mlp.keywords["submodules"].activation_func = XIELU
    else:
        mlp.submodules.activation_func = ModuleSpec(module=XIELU)
    return spec


@dataclass
class ApertusModelProvider(GPTModelProvider):
    """Configuration class for Apertus models.

    Apertus is based on Llama architecture with:
    - XIELU activation (learnable alpha_p, alpha_n; NOT a gated MLP)
    - QK normalization
    - Llama3-style RoPE scaling (native mcore rope_scaling passthrough)
    """

    # Core architecture settings (Llama-like but NOT gated MLP)
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = False
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False

    # Apertus-specific: QK normalization
    qk_layernorm: bool = True

    # XIELU is built from the layer spec as a module activation.
    # Llama3-style RoPE scaling uses the parent's native rope_scaling /
    # rope_scaling_factor fields, populated from the HF config by the bridge.
    use_te_activation_func: bool = True
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTModelProvider"], ModuleSpec]] = apertus_layer_spec

    # Fusions — bias_activation_fusion must stay off for a module activation
    bias_activation_fusion: bool = False
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = True

    def finalize(self) -> None:
        """Validate the config, enforcing the module-activation fusion invariant.

        Downstream config systems (e.g. NeMo-RL's megatron_cfg) overwrite
        fusion flags after construction and before finalize();
        bias_activation_fusion cannot apply to a module activation, so enforce
        the invariant here instead of relying on every recipe to override it.
        """
        if self.use_te_activation_func and self.bias_activation_fusion:
            logger.warning(
                "Apertus: forcing bias_activation_fusion=False — the fusion is "
                "incompatible with the XIELU module activation."
            )
            self.bias_activation_fusion = False
        super().finalize()


@dataclass
class ApertusModelProvider8B(ApertusModelProvider):
    """Configuration for the public Apertus-8B model (swiss-ai/Apertus-8B-2509).

    Architecture:
    - 32 layers, 4096 hidden size
    - 32 attention heads, 8 KV heads (GQA)
    - 21504 FFN hidden size (ungated)
    - 131072 vocab size
    - 65536 max position embeddings (8x llama3 RoPE scaling from 8192)
    - 12M rope_theta
    """

    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    ffn_hidden_size: int = 21504
    seq_length: int = 65536
    rotary_base: float = 12_000_000.0
    rope_scaling: bool = True
    rope_scaling_factor: float = 8.0
