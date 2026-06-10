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

"""Bridge-owned XIELU activation for Apertus models.

Math is identical to the Swiss AI Megatron-LM implementation
(megatron/core/activations.py): alpha parameters are stored raw
(pre-softplus, matching the checkpoint layout), softplus is applied in
forward, and beta is added to the negative-branch coefficient.

Forward dispatches to the fused CUDA kernel from the optional ``xielu``
extension when the input qualifies (CUDA, bf16, numel divisible by 128),
otherwise falls back to a jit-fused eager implementation. Set
``APERTUS_EAGER_XIELU=1`` to force the eager path for debugging.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule

try:
    from xielu import xielu as _xielu_cuda
except ImportError:
    _xielu_cuda = None


@jit_fuser
def compiled_xielu(x, alpha_p, alpha_n, beta: float, eps: float):
    """Eager XIELU on post-softplus coefficients (Swiss-fork formula)."""
    return torch.where(
        x > 0,
        alpha_p * x * x + beta * x,
        alpha_n * torch.expm1(torch.clamp(x, max=eps)) - alpha_n * x + beta * x,
    )


class XIELU(MegatronModule):
    """XIELU activation with learnable alpha_p/alpha_n stored raw (pre-softplus).

    Instantiated per-MLP via the layer spec (``MLPSubmodules.activation_func``)
    so alpha_p/alpha_n appear as ``mlp.activation_func.*`` parameters, matching
    both the HF checkpoint layout and the Swiss-fork Megatron param names.
    """

    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6, dtype=None):
        super().__init__(config=config)
        if dtype is None:
            dtype = getattr(config, "params_dtype", None) or torch.float32
        self.alpha_p = nn.Parameter(
            torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=dtype)) - 1.0).unsqueeze(0)
        )
        self.alpha_n = nn.Parameter(
            torch.log(torch.exp(torch.tensor(alpha_n_init - beta, dtype=dtype)) - 1.0).unsqueeze(0)
        )
        self.beta = beta
        self.eps = eps
        self._force_eager = os.environ.get("APERTUS_EAGER_XIELU", "0") == "1"

    def _cuda_usable(self, x: torch.Tensor) -> bool:
        return (
            _xielu_cuda is not None
            and not self._force_eager
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and self.alpha_p.dtype == torch.bfloat16
            and x.numel() % 128 == 0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cuda_usable(x):
            return _xielu_cuda(x, self.alpha_p, self.alpha_n, self.beta, self.eps)
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        return compiled_xielu(x, alpha_p, alpha_n, self.beta, self.eps)
