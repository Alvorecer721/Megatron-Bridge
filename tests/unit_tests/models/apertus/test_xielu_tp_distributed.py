# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two-rank regression test for Apertus XIELU tensor-parallel training.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/models/apertus/test_xielu_tp_distributed.py
"""

import os

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from megatron.bridge.models.apertus.xielu_activation import XIELU, compiled_xielu


_TP_SIZE = 2


@pytest.mark.gpu
def test_xielu_tp2_sums_alpha_gradients_and_keeps_replicas_synced() -> None:
    """Replicated XIELU alphas must receive summed gradients and remain synchronized."""
    if int(os.environ.get("WORLD_SIZE", "1")) != _TP_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    owns_process_group = not dist.is_initialized()
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    if owns_process_group:
        dist.init_process_group(backend="nccl")

    try:
        if owns_model_parallel:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=_TP_SIZE,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
            )

        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_rank = dist.get_rank(group=tp_group)
        unfixed = XIELU(config=None, dtype=torch.float32).cuda()
        fixed = XIELU(config=None, dtype=torch.float32).cuda()
        initial_alpha_p = fixed.alpha_p.detach().clone()
        initial_alpha_n = fixed.alpha_n.detach().clone()

        # Mimic different FC1 output shards: both ranks contribute different
        # positive- and negative-branch gradients to the shared scalars.
        inputs = (
            torch.tensor([-1.0, -0.25, 0.5, 1.5], device="cuda"),
            torch.tensor([-3.0, -0.75, 2.0, 4.0], device="cuda"),
        )[tp_rank]

        local_expected = torch.stack(
            (
                initial_alpha_p.sigmoid() * inputs[inputs > 0].square().sum(),
                initial_alpha_n.sigmoid()
                * (torch.expm1(inputs[inputs <= 0].clamp(max=fixed.eps)) - inputs[inputs <= 0]).sum(),
            )
        ).flatten()
        global_expected = local_expected.clone()
        dist.all_reduce(global_expected, op=dist.ReduceOp.SUM, group=tp_group)

        unfixed_optimizer = torch.optim.SGD(unfixed.parameters(), lr=0.1)
        fixed_optimizer = torch.optim.SGD(fixed.parameters(), lr=0.1)
        compiled_xielu(inputs, unfixed.alpha_p, unfixed.alpha_n, unfixed.beta, unfixed.eps).sum().backward()
        fixed(inputs).sum().backward()

        torch.testing.assert_close(unfixed.alpha_p.grad, local_expected[0].reshape_as(unfixed.alpha_p))
        torch.testing.assert_close(unfixed.alpha_n.grad, local_expected[1].reshape_as(unfixed.alpha_n))
        torch.testing.assert_close(fixed.alpha_p.grad, global_expected[0].reshape_as(fixed.alpha_p))
        torch.testing.assert_close(fixed.alpha_n.grad, global_expected[1].reshape_as(fixed.alpha_n))

        unfixed_optimizer.step()
        fixed_optimizer.step()
        expected_alpha_p = initial_alpha_p - 0.1 * global_expected[0]
        expected_alpha_n = initial_alpha_n - 0.1 * global_expected[1]
        torch.testing.assert_close(fixed.alpha_p, expected_alpha_p.reshape_as(fixed.alpha_p))
        torch.testing.assert_close(fixed.alpha_n, expected_alpha_n.reshape_as(fixed.alpha_n))

        spreads = {}
        for label, module in (("before_fix", unfixed), ("after_fix", fixed)):
            gradients = torch.stack((module.alpha_p.grad, module.alpha_n.grad))
            parameters = torch.stack((module.alpha_p.detach(), module.alpha_n.detach()))
            gathered_gradients = [torch.empty_like(gradients) for _ in range(_TP_SIZE)]
            gathered_parameters = [torch.empty_like(parameters) for _ in range(_TP_SIZE)]
            dist.all_gather(gathered_gradients, gradients, group=tp_group)
            dist.all_gather(gathered_parameters, parameters, group=tp_group)
            spreads[label] = (
                (gathered_gradients[0] - gathered_gradients[1]).abs().flatten(),
                (gathered_parameters[0] - gathered_parameters[1]).abs().flatten(),
            )

        assert torch.all(spreads["before_fix"][0] > 0)
        assert torch.all(spreads["before_fix"][1] > 0)
        torch.testing.assert_close(spreads["after_fix"][0], torch.zeros(2, device="cuda"))
        torch.testing.assert_close(spreads["after_fix"][1], torch.zeros(2, device="cuda"))
    finally:
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
