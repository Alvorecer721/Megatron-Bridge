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

Apertus uses XIELU activation which has learnable parameters (alpha_p, alpha_n).
Requires Swiss AI Megatron-LM in PYTHONPATH for the XIELU implementation.

Usage:
    export PYTHONPATH=/path/to/swiss-ai-megatron-lm:$PYTHONPATH
    python convert_checkpoints.py import --hf-model swiss-ai/Apertus-8B-2509 ...
"""

import logging
from dataclasses import dataclass
from typing import Optional

from megatron.bridge.models.gpt_provider import GPTModelProvider

# Import XIELU - works with both stock megatron-lm and Swiss AI Megatron-LM
try:
    from megatron.training.activations import XIELU
    HAS_XIELU = True
except ImportError:
    HAS_XIELU = False
    XIELU = None

logger = logging.getLogger(__name__)


@dataclass
class ApertusModelProvider(GPTModelProvider):
    """Configuration class for Apertus models (swiss-ai/Apertus-8B-2509).

    Apertus is based on Llama architecture with:
    - XIELU activation function (learnable alpha_p, alpha_n parameters)
    - QK normalization enabled
    - Llama3-style RoPE scaling (8x from 8K to 64K context)

    Requires Swiss AI Megatron-LM in PYTHONPATH for XIELU support.
    """

    # Core architecture settings (Llama-like but NOT gated MLP)
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = False  # Apertus uses XIELU activation, NOT gated MLP like SwiGLU
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False

    # Apertus-specific: QK normalization
    qk_layernorm: bool = True

    # RoPE scaling factor (Llama3-style)
    scale_factor: float = 8.0

    # Fusions - disable bias_activation_fusion since XIELU is custom
    bias_activation_fusion: bool = False
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = True
    use_transformer_engine_op_fuser: Optional[bool] = None

    def __post_init__(self):
        super().__post_init__()

        # Set XIELU activation
        if HAS_XIELU:
            self.activation_func = XIELU
            logger.info("Using XIELU activation function for Apertus model")
        else:
            raise ImportError(
                "XIELU activation not found. Please ensure Swiss AI Megatron-LM is in PYTHONPATH:\n"
                "export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/Megatron-LM:$PYTHONPATH"
            )

        # Set RoPE scaling factor for Llama3-style scaling
        self.rotary_scaling_factor = self.scale_factor


@dataclass
class ApertusModelProvider8B(ApertusModelProvider):
    """Configuration for Apertus-8B model (swiss-ai/Apertus-8B-2509).

    Architecture:
    - 32 layers
    - 4096 hidden size
    - 32 attention heads, 8 KV heads (GQA)
    - 21504 FFN hidden size
    - 131072 vocab size
    - 65536 max position embeddings
    - 12M rope_theta
    """

    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    ffn_hidden_size: int = 21504
    seq_length: int = 65536
    rotary_base: float = 12000000.0
