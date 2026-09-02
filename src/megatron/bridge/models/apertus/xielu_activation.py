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

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from megatron.core.transformer.module import MegatronModule


logger = logging.getLogger(__name__)

try:
    from xielu import xielu as _xielu_cuda
except ImportError:
    _xielu_cuda = None


@jit_fuser
def compiled_xielu(x, raw_alpha_p, raw_alpha_n, beta: float, eps: float):
    """Eager XIELU on raw (pre-softplus) alphas (Swiss-fork formula).

    The softplus reparameterization lives inside the fused region so the
    scalar ops fold into the elementwise kernel instead of launching
    separately per call.
    """
    alpha_p = F.softplus(raw_alpha_p)
    alpha_n = beta + F.softplus(raw_alpha_n)
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

    def __init__(
        self,
        config=None,
        alpha_p_init=0.8,
        alpha_n_init=0.8,
        beta=0.5,
        eps=-1e-6,
        dtype=None,
    ):
        super().__init__(config=config)
        if dtype is None:
            dtype = getattr(config, "params_dtype", None) or torch.float32
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=dtype)) - 1.0).unsqueeze(0))
        self.alpha_n = nn.Parameter(
            torch.log(torch.exp(torch.tensor(alpha_n_init - beta, dtype=dtype)) - 1.0).unsqueeze(0)
        )
        self.beta = beta
        self.eps = eps
        self._force_eager = os.environ.get("APERTUS_EAGER_XIELU", "0") == "1"

    # class-level so each dispatch path is logged once per process, not per layer
    _logged_paths: set = set()

    @classmethod
    def _log_once(cls, key: str, msg: str) -> None:
        if key not in cls._logged_paths:
            cls._logged_paths.add(key)
            # eager fallback is a silent performance degradation -> WARNING so
            # it surfaces even under Ray workers' default log level
            (logger.warning if key == "eager" else logger.info)(msg)

    def _cuda_usable(self, x: torch.Tensor) -> bool:
        return (
            _xielu_cuda is not None
            and not self._force_eager
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and self.alpha_p.dtype == torch.bfloat16
            and x.numel() % 128 == 0
        )

    def _eager_reason(self, x: torch.Tensor) -> str:
        # diagnostic mirror of _cuda_usable, evaluated once per process when
        # the fallback first fires — keep the conditions in sync
        if _xielu_cuda is None:
            return "xielu CUDA extension not installed"
        if self._force_eager:
            return "APERTUS_EAGER_XIELU=1"
        if not x.is_cuda:
            return "input not on CUDA"
        if x.dtype != torch.bfloat16 or self.alpha_p.dtype != torch.bfloat16:
            return f"dtype {x.dtype}/{self.alpha_p.dtype} (kernel is bf16-only)"
        return f"numel {x.numel()} not divisible by 128"

    def _tensor_parallel_alphas(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return replicated alphas whose backward gradients are TP-summed.

        FC1 partitions Apertus's FFN activations over tensor-parallel ranks, so
        each rank computes only part of each shared alpha's gradient. MCore's
        copy-to-TP autograd mapping is an identity in forward and a SUM
        all-reduce in backward, producing the full gradient on every replica.
        """
        if not parallel_state.model_parallel_is_initialized():
            return self.alpha_p, self.alpha_n
        tp_group = parallel_state.get_tensor_model_parallel_group()
        return (
            copy_to_tensor_model_parallel_region(self.alpha_p, group=tp_group),
            copy_to_tensor_model_parallel_region(self.alpha_n, group=tp_group),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p, alpha_n = self._tensor_parallel_alphas()
        if self._cuda_usable(x):
            self._log_once("cuda", "Apertus XIELU: using fused CUDA kernel (xielu extension)")
            return _xielu_cuda(x, alpha_p, alpha_n, self.beta, self.eps)
        if "eager" not in XIELU._logged_paths:  # build the reason string only on first fallback
            self._log_once(
                "eager",
                f"Apertus XIELU: using eager fallback ({self._eager_reason(x)})",
            )
        return compiled_xielu(x, alpha_p, alpha_n, self.beta, self.eps)
