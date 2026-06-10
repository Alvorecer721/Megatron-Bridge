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

"""Apertus Model Bridge for converting between HuggingFace and Megatron formats.

Apertus is a Swiss AI model based on Llama architecture with:
- XIELU activation (learnable alpha_p, alpha_n parameters)
- QK normalization
- Llama3-style RoPE scaling

Usage:
    export PYTHONPATH=/path/to/swiss-ai-megatron-lm:$PYTHONPATH
    python convert_checkpoints.py import --hf-model swiss-ai/Apertus-8B-2509 --trust-remote-code ...
"""

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    QKVMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.apertus.apertus_provider import ApertusModelProvider

# Apertus uses a custom model class that requires trust_remote_code
# We register by model name string to avoid loading the model at import time
# String-based registration matches the HF model's architecture name

@MegatronModelBridge.register_bridge(source="ApertusForCausalLM", target=GPTModel)
class ApertusBridge(MegatronModelBridge):
    """
    Megatron Bridge for Apertus Causal LM.

    Converts between HuggingFace swiss-ai/Apertus-8B and Megatron GPT format.

    Key differences from LlamaBridge:
    - Maps XIELU activation parameters (alpha_p, alpha_n)
    - Enables QK normalization
    - Uses Llama3-style RoPE scaling

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained(
        ...     "swiss-ai/Apertus-8B-2509",
        ...     trust_remote_code=True
        ... )
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> ApertusModelProvider:
        hf_config = hf_pretrained.config

        # Llama3-style RoPE scaling: read the full dict; absent means the
        # checkpoint uses unscaled RoPE (scale_factor=None -> no scaling).
        rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
        scale_factor = rope_scaling.get("factor")
        low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
        old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

        # Extract kv_channels from head_dim if present
        kv_channels = getattr(hf_config, "head_dim", None)

        provider = ApertusModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            num_query_groups=hf_config.num_key_value_heads,
            seq_length=hf_config.max_position_embeddings,
            rotary_base=hf_config.rope_theta,
            kv_channels=kv_channels,
            gated_linear_unit=False,  # Apertus uses XIELU, NOT gated MLP
            # Apertus-specific: QK normalization
            qk_layernorm=getattr(hf_config, "qk_norm", True),
            # RoPE scaling (full llama3 parameterization from HF config)
            scale_factor=scale_factor,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            old_context_len=old_context_len,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
            vocab_size=hf_config.vocab_size,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return parameter mappings from Megatron to HF format.

        Apertus uses different layernorm names and XIELU activation (not gated MLP).
        """
        # Register XIELU as replicated module type (learnable but not tensor-parallel)
        AutoMapping.register_module_type('XIELU', 'replicated')

        # Apertus HF parameter names are different from standard Llama:
        # - attention_layernorm instead of input_layernorm
        # - feedforward_layernorm instead of post_attention_layernorm
        # - q_norm/k_norm for QK normalization
        # - No gate_proj (not gated MLP), just up_proj with XIELU activation
        param_mappings = {
            # Embeddings and output
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",

            # ======================================================================
            # Layer norms - Swiss AI / older megatron-core (separate layernorm modules)
            # Use these mappings when running with Swiss AI Megatron-LM fork
            # ======================================================================
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.attention_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.feedforward_layernorm.weight",

            # ======================================================================
            # Layer norms - newer megatron-core (fused layernorm in linear layers)
            # Use these mappings when running with /opt/megatron-lm or newer megatron-core
            # After conversion, run rename_checkpoint_keys.py to convert to Swiss AI format
            # ======================================================================
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.feedforward_layernorm.weight",

            # Attention output projection
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",

            # MLP (NOT gated - Apertus uses XIELU activation on up_proj directly)
            "decoder.layers.*.mlp.linear_fc1.weight": "model.layers.*.mlp.up_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",

            # QK normalization weights (Apertus-specific)
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",

            # XIELU activation parameters (Apertus-specific)
            "decoder.layers.*.mlp.activation_func.alpha_p": "model.layers.*.mlp.act_fn.alpha_p",
            "decoder.layers.*.mlp.activation_func.alpha_n": "model.layers.*.mlp.act_fn.alpha_n",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add QKV mapping (combine separate Q, K, V into single QKV matrix)
        mapping_list.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
        )
        # Note: No GatedMLPMapping - Apertus uses simple up_proj + XIELU, not gate_proj * up_proj

        return MegatronMappingRegistry(*mapping_list)
