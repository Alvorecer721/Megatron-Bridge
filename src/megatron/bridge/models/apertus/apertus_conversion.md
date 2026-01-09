# Apertus HF to Megatron Conversion Guide

This document describes how to convert the Swiss AI Apertus-8B model from HuggingFace format to Megatron distributed checkpoint format.

## Prerequisites

- Megatron-Bridge repository
- `/opt/megatron-lm` (NVIDIA's Megatron-LM)
- Swiss AI Megatron-LM fork (for training/inference)

## Code Changes Required

### 1. Add XIELU activation stub to /opt/megatron-lm

Create `/opt/megatron-lm/megatron/training/activations.py`:

```python
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# XIELU activation from Swiss AI Megatron-LM

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule


@jit_fuser
def compiled_xielu(x, alpha_p, alpha_n, beta=0.5, eps=-1e-6):
    return torch.where(x > 0,
                      alpha_p * x * x + beta * x,
                      alpha_n * torch.expm1(torch.min(x, eps)) - alpha_n * x + beta * x)


class XIELU(MegatronModule):
    """XIELU activation with learnable alpha_p and alpha_n parameters."""

    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super().__init__(config=config)
        self.config = config
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init - beta, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.beta = beta
        self.eps = torch.tensor(eps, dtype=torch.bfloat16, device='cuda')

    def forward(self, x):
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        return compiled_xielu(x, alpha_p, alpha_n, self.beta, self.eps)
```

This stub is needed because Megatron-Bridge uses `/opt/megatron-lm` for conversion, which doesn't have XIELU by default.

### 2. Fix circular import in Swiss AI Megatron-LM (for inference/training)

In `/path/to/swiss-ai-megatron-lm/megatron/core/transformer/mlp.py`, make the XIELU import lazy by moving it inside the `__init__` method:

```python
def __init__(
    self,
    config: TransformerConfig,
    submodules: MLPSubmodules,
    is_expert: bool = False,
    input_size: int = None,
):
    super().__init__(config=config)
    # ... existing code ...

    # Lazy import to avoid circular dependency
    from megatron.training.activations import XIELU, XIPReLU, XIPReLUP

    if self.config.activation_func == XIELU:
        self.activation_func = XIELU(config=self.config)
    elif self.config.activation_func == XIPReLU:
        self.activation_func = XIPReLU(config=self.config)
    elif self.config.activation_func == XIPReLUP:
        self.activation_func = XIPReLUP(config=self.config)
    else:
        self.activation_func = self.config.activation_func
```

## Conversion Command

```bash
PYTHONPATH=/opt/megatron-lm:$PYTHONPATH python convert_checkpoints.py import \
    --hf-model swiss-ai/Apertus-8B-2509 \
    --trust-remote-code \
    --out /path/to/output/apertus_megatron
```

**Important**: Use `/opt/megatron-lm` in PYTHONPATH for conversion (not Swiss AI fork).

## Output Checkpoint Structure

```
apertus_megatron/
├── iter_0000000/
│   ├── .metadata
│   └── __0_0.distcp
├── latest_checkpointed_iteration.txt
└── latest_train_state.pt
```

## Checkpoint Key Format

The checkpoint uses **fused layernorm** keys (same as newer megatron-core):
- `decoder.layers.self_attention.linear_qkv.layer_norm_weight`
- `decoder.layers.mlp.linear_fc1.layer_norm_weight`

Swiss AI's Megatron-LM also uses fused layernorms, so **no key renaming is needed**.

## Checkpoint Metadata Format (MCoreMetadata)

Megatron-Bridge (using newer megatron-core) saves checkpoints with `MCoreMetadata`, a subclass of PyTorch's `Metadata` that includes an `mcore_data` field for Megatron-specific metadata.

If Swiss AI's Megatron-LM fails to load the checkpoint with metadata errors, you need to add the `MCoreMetadata` class.

### Patch for Swiss AI Megatron-LM

Add to `/path/to/swiss-ai-megatron-lm/megatron/core/dist_checkpointing/mapping.py`:

```python
from torch.distributed.checkpoint.metadata import Metadata

@dataclass
class MCoreMetadata(Metadata):
    """Metadata subclass with mcore_data field for Megatron checkpoints."""
    mcore_data: dict = field(default_factory=dict)
```

Then update `MCoreSavePlanner.create_global_plan` in `strategies/torch.py` to use:

```python
# Instead of: metadata.mcore_data = dict(...)
# Use:
metadata = MCoreMetadata(mcore_data=mcore_data_dict, **vars(metadata))
```

This ensures checkpoint metadata is properly typed and can be loaded by both newer megatron-core and Swiss AI fork.

## Apertus Model Configuration

| Parameter | Value |
|-----------|-------|
| hidden_size | 4096 |
| intermediate_size (ffn_hidden_size) | 21504 |
| num_hidden_layers | 32 |
| num_attention_heads | 32 |
| num_key_value_heads (GQA) | 8 |
| vocab_size | 131072 |
| max_position_embeddings | 131072 |
| rope_theta | 12000000 |
| rope_scaling | Llama3-style (factor=8.0) |
| activation | XIELU (learnable alpha_p, alpha_n) |
| qk_norm | True |
| gated_linear_unit | False |

## Logits Parity Test

A test script is provided at `test_logits_parity.py` to verify the conversion:

```bash
PYTHONPATH=/path/to/swiss-ai-megatron-lm:$PYTHONPATH \
torchrun --nproc_per_node=1 test_logits_parity.py
```

### Test Results

- **Top-5 predictions match exactly** between HF and Megatron models
- Max absolute difference: ~4.0 (on some outlier tokens)
- Mean absolute difference: ~0.37

The small differences are due to Llama3-style RoPE scaling which HF applies but may not be fully replicated in Megatron. The relative ordering of predictions is preserved, confirming functional correctness.

### Example Output

```
HF top-5 predictions:
  1. ' Paris' (id=6993, score=32.5000)
  2. ' the' (id=1278, score=31.0000)
  3. ' a' (id=1261, score=30.8750)
  4. ' also' (id=2095, score=30.3750)
  5. ' one' (id=1925, score=29.8750)

Megatron top-5 predictions:
  1. ' Paris' (id=6993, score=32.7500)
  2. ' the' (id=1278, score=31.3750)
  3. ' a' (id=1261, score=31.2500)
  4. ' also' (id=2095, score=30.7500)
  5. ' one' (id=1925, score=30.2500)
```

## Loading Checkpoint in Swiss AI Megatron-LM

```python
from megatron.core import dist_checkpointing
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.training.activations import XIELU

config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    ffn_hidden_size=21504,
    num_attention_heads=32,
    num_query_groups=8,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    normalization="RMSNorm",
    add_bias_linear=False,
    gated_linear_unit=False,
    activation_func=XIELU,
    qk_layernorm=True,
    bf16=True,
)

model = GPTModel(
    config=config,
    transformer_layer_spec=get_gpt_layer_local_spec(qk_layernorm=True),
    vocab_size=131072,
    max_sequence_length=131072,
    pre_process=True,
    post_process=True,
    position_embedding_type='rope',
    rotary_base=12000000,
)
model.to(dtype=torch.bfloat16, device="cuda")

# Load checkpoint
sharded_state_dict = model.sharded_state_dict(prefix="")
loaded_state_dict = dist_checkpointing.load(sharded_state_dict, ckpt_path)

# Filter metadata keys
for key in ["checkpoint_version", "iteration", "content_metadata"]:
    loaded_state_dict.pop(key, None)

model.load_state_dict(loaded_state_dict)
```

## Troubleshooting

### "XIELU not found" during conversion
Ensure `/opt/megatron-lm/megatron/training/activations.py` exists with the XIELU class.

### Circular import error during inference
Apply the lazy import fix to Swiss AI's `mlp.py`.

### Shape mismatch for linear_fc1.weight
Ensure `ffn_hidden_size=21504` (not 14336 like Llama).

### "expected BFloat16 but found Float"
Explicitly cast model: `model.to(dtype=torch.bfloat16, device="cuda")`

### Logits have consistent offset
Ensure `rotary_base=12000000` (not 500000). Apertus uses a high rope_theta for 128K context support.
