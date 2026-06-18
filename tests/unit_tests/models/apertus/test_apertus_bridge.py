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

from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.apertus.apertus_bridge import ApertusBridge
from megatron.bridge.models.apertus.xielu_activation import XIELU
from megatron.bridge.models.conversion.model_bridge import WeightConversionTask


def _alpha_task(module: XIELU, global_param_name: str = "decoder.layers.3.mlp.activation_func.alpha_p"):
    return WeightConversionTask(
        param_name=global_param_name.rsplit(".", 1)[-1],
        global_param_name=global_param_name,
        mapping=Mock(),
        megatron_module=module,
        param_weight=module.alpha_p,
    )


def test_export_injects_xielu_beta_eps_with_alpha_dtype_and_device():
    bridge = ApertusBridge()
    module = XIELU(config=None, dtype=torch.bfloat16)
    alpha = torch.ones(1, dtype=torch.bfloat16)
    converted = {"model.layers.3.mlp.act_fn.alpha_p": alpha}

    result = bridge.maybe_modify_converted_hf_weight(_alpha_task(module), dict(converted), {})
    expected_beta = torch.tensor(module.beta, dtype=alpha.dtype).item()
    expected_eps = torch.tensor(module.eps, dtype=alpha.dtype).item()

    assert result["model.layers.3.mlp.act_fn.alpha_p"] is alpha
    assert result["model.layers.3.mlp.act_fn.beta"].dtype == alpha.dtype
    assert result["model.layers.3.mlp.act_fn.beta"].device == alpha.device
    assert result["model.layers.3.mlp.act_fn.beta"].item() == pytest.approx(expected_beta)
    assert result["model.layers.3.mlp.act_fn.eps"].dtype == alpha.dtype
    assert result["model.layers.3.mlp.act_fn.eps"].device == alpha.device
    assert result["model.layers.3.mlp.act_fn.eps"].item() == pytest.approx(expected_eps)


def test_export_does_not_inject_xielu_beta_eps_for_alpha_n():
    bridge = ApertusBridge()
    module = XIELU(config=None, dtype=torch.float32)
    task = _alpha_task(module, "decoder.layers.3.mlp.activation_func.alpha_n")
    converted = {"model.layers.3.mlp.act_fn.alpha_n": module.alpha_n}

    result = bridge.maybe_modify_converted_hf_weight(task, dict(converted), {})

    assert "model.layers.3.mlp.act_fn.beta" not in result
    assert "model.layers.3.mlp.act_fn.eps" not in result


def test_export_reports_clear_error_when_alpha_p_conversion_is_missing():
    bridge = ApertusBridge()
    module = XIELU(config=None, dtype=torch.float32)
    converted = {"model.layers.3.mlp.act_fn.unexpected": module.alpha_p}

    with pytest.raises(ValueError, match="alpha_p"):
        bridge.maybe_modify_converted_hf_weight(_alpha_task(module), converted, {})
