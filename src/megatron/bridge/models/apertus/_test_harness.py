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

"""Shared harness for the standalone Apertus test scripts (same directory)."""

import os
import sys

import torch

failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    if not cond:
        failures.append(name)


def finish():
    print(f"\n{len(failures)} failure(s)" if failures else "\nALL CHECKS PASSED", flush=True)
    sys.exit(1 if failures else 0)


def dist_init(port=29511):
    """Single-rank NCCL + megatron parallel state, for 1-GPU model builds."""
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl", world_size=1, rank=0)
    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(42)
