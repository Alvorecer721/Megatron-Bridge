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

"""Unit tests for Apertus bridge internals: XIELU module, MLP wiring, RoPE scaling.

Requires 1 GPU and stock megatron-core (no Swiss fork). Run standalone:
    PYTHONPATH=<bridge>/src[:<xielu-site>] python test_apertus_provider.py
Exit code 0 = all checks pass.
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

BETA, EPS = 0.5, -1e-6
failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


def eager_ref_fp32(x, raw_ap, raw_an, beta=BETA, eps=EPS):
    """Swiss-fork XIELU reference in fp32 (megatron/core/activations.py)."""
    x = x.float()
    ap = F.softplus(raw_ap.float())
    an = beta + F.softplus(raw_an.float())
    return torch.where(x > 0, ap * x * x + beta * x, an * torch.expm1(torch.clamp(x, max=eps)) - an * x + beta * x)


def test_xielu_module():
    from megatron.bridge.models.apertus.xielu_activation import XIELU

    m = XIELU(config=None, dtype=torch.bfloat16).cuda()
    params = dict(m.named_parameters())
    check("xielu has alpha params", set(params) == {"alpha_p", "alpha_n"}, f"got {sorted(params)}")
    check("xielu params dtype", all(p.dtype == torch.bfloat16 for p in params.values()))

    g = torch.Generator(device="cuda").manual_seed(5)
    x = (torch.randn(64, 256, generator=g, device="cuda") * 2).bfloat16().requires_grad_(True)
    out = m(x)
    ref = eager_ref_fp32(x.detach(), m.alpha_p.detach(), m.alpha_n.detach())
    ok = torch.allclose(out.float(), ref, rtol=0.03, atol=0.03)
    check("xielu forward parity", ok, f"max_err={(out.float() - ref).abs().max().item():.3e}")

    out.sum().backward()
    check(
        "xielu alpha grads flow",
        m.alpha_p.grad is not None and m.alpha_n.grad is not None and x.grad is not None,
    )


def _dist_init():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl", world_size=1, rank=0)
    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(42)


def _tiny_provider(**overrides):
    from megatron.bridge.models.apertus import ApertusModelProvider

    kwargs = dict(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=32,
        seq_length=128,
        vocab_size=256,
        make_vocab_size_divisible_by=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        rotary_base=4_000_000.0,
        # bare unit test: no DDP, so no main_grad buffers for the fused path
        gradient_accumulation_fusion=False,
    )
    kwargs.update(overrides)
    return ApertusModelProvider(**kwargs)


def test_provider_builds_on_stock_mcore():
    p = _tiny_provider()
    if hasattr(p, "finalize"):
        p.finalize()
    model = p.provide(pre_process=True, post_process=True).cuda()

    names = dict(model.named_parameters())
    has_alpha = "decoder.layers.0.mlp.activation_func.alpha_p" in names
    check(
        "mlp.activation_func.alpha_p exists (spec wiring)",
        has_alpha,
        "" if has_alpha else f"mlp params: {[n for n in names if 'mlp' in n][:6]}",
    )

    ids = torch.randint(0, 256, (1, 32), device="cuda")
    pos = torch.arange(32, device="cuda").unsqueeze(0)
    out = model(input_ids=ids, position_ids=pos, attention_mask=None)
    out.float().sum().backward()
    ap = names.get("decoder.layers.0.mlp.activation_func.alpha_p")
    check("alpha grads flow through model", ap is not None and ap.grad is not None and ap.grad.abs().sum() > 0)
    return model


def test_rope_scaling_applied():
    p0 = _tiny_provider()  # rope_scaling defaults to False
    p1 = _tiny_provider(rope_scaling=True, rope_scaling_factor=32.0)
    for p in (p0, p1):
        if hasattr(p, "finalize"):
            p.finalize()
    m0 = p0.provide(pre_process=True, post_process=True)
    m1 = p1.provide(pre_process=True, post_process=True)

    base = m0.rotary_pos_emb.inv_freq
    got = m1.rotary_pos_emb.inv_freq
    # llama3 scaling only ever reduces frequencies: low-freq components are
    # divided by the full factor, high-freq are untouched, mid are smoothed.
    ratio = got / base
    check("rope_scaling=False leaves inv_freq unscaled", not torch.equal(base, got))
    check(
        "factor=32 scaling shape (low/32 .. high unchanged)",
        torch.isclose(ratio.min(), torch.tensor(1.0 / 32.0, device=ratio.device), rtol=1e-4).item()
        and torch.isclose(ratio.max(), torch.tensor(1.0, device=ratio.device), rtol=1e-6).item()
        and bool((ratio <= 1.0 + 1e-6).all()),
        f"ratio range=({ratio.min().item():.5f}, {ratio.max().item():.5f})",
    )


def test_bridge_parses_full_rope_dict():
    from megatron.bridge.models.apertus.apertus_bridge import ApertusBridge

    def fake_hf(rope_scaling):
        cfg = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            num_key_value_heads=4,
            max_position_embeddings=128,
            rope_theta=4_000_000,
            rope_scaling=rope_scaling,
            qk_norm=True,
            vocab_size=256,
            tie_word_embeddings=False,
            torch_dtype=torch.bfloat16,
        )
        return SimpleNamespace(config=cfg, generation_config=None)

    b = ApertusBridge()
    apertus15 = {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    }
    p = b.provider_bridge(fake_hf(apertus15))
    check(
        "bridge: factor=32 parsed",
        getattr(p, "rope_scaling", None) is True and getattr(p, "rope_scaling_factor", None) == 32.0,
        f"got rope_scaling={getattr(p, 'rope_scaling', None)} factor={getattr(p, 'rope_scaling_factor', None)}",
    )

    p_none = b.provider_bridge(fake_hf(None))
    check("bridge: rope_scaling=None -> no scaling", getattr(p_none, "rope_scaling", "unset") is False)

    # non-default llama3 params must be rejected loudly, not silently dropped
    bad = dict(apertus15, low_freq_factor=2.0)
    try:
        b.provider_bridge(fake_hf(bad))
        check("bridge: non-default rope params rejected", False, "accepted silently")
    except ValueError as e:
        check("bridge: non-default rope params rejected", True, str(e)[:60])


def main():
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    test_xielu_module()
    _dist_init()
    test_provider_builds_on_stock_mcore()
    test_rope_scaling_applied()
    test_bridge_parses_full_rope_dict()

    print(f"\n{len(failures)} failure(s)" if failures else "\nALL APERTUS PROVIDER CHECKS PASSED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
