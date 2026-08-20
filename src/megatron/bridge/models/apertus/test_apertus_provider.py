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
    PYTHONPATH=<bridge>/src[:<xielu-site>] python test_apertus_provider.py [tokenizer_dir]
The optional tokenizer_dir argument additionally asserts the tokenizer loads
and carries a chat template (used by job prologues). Exit code 0 = all pass.
"""

import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from _test_harness import check, dist_init, finish


BETA, EPS = 0.5, -1e-6


def eager_ref_fp32(x, raw_ap, raw_an, beta=BETA, eps=EPS):
    """Swiss-fork XIELU reference in fp32 (megatron/core/activations.py)."""
    x = x.float()
    ap = F.softplus(raw_ap.float())
    an = beta + F.softplus(raw_an.float())
    return torch.where(
        x > 0,
        ap * x * x + beta * x,
        an * torch.expm1(torch.clamp(x, max=eps)) - an * x + beta * x,
    )


def test_xielu_module():
    from megatron.bridge.models.apertus.xielu_activation import XIELU

    m = XIELU(config=None, dtype=torch.bfloat16).cuda()
    params = dict(m.named_parameters())
    check(
        "xielu has alpha params",
        set(params) == {"alpha_p", "alpha_n"},
        f"got {sorted(params)}",
    )
    check("xielu params dtype", all(p.dtype == torch.bfloat16 for p in params.values()))

    g = torch.Generator(device="cuda").manual_seed(5)
    x = (torch.randn(64, 256, generator=g, device="cuda") * 2).bfloat16().requires_grad_(True)
    out = m(x)
    ref = eager_ref_fp32(x.detach(), m.alpha_p.detach(), m.alpha_n.detach())
    ok = torch.allclose(out.float(), ref, rtol=0.03, atol=0.03)
    check(
        "xielu forward parity",
        ok,
        f"max_err={(out.float() - ref).abs().max().item():.3e}",
    )

    out.sum().backward()
    check(
        "xielu alpha grads flow",
        m.alpha_p.grad is not None and m.alpha_n.grad is not None and x.grad is not None,
    )


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
    provider = ApertusModelProvider(**kwargs)
    provider.finalize()
    return provider


def test_provider_builds_on_stock_mcore():
    model = _tiny_provider().provide(pre_process=True, post_process=True).cuda()

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
    check(
        "alpha grads flow through model",
        ap is not None and ap.grad is not None and ap.grad.abs().sum() > 0,
    )


def test_fusion_invariant_enforced():
    # NeMo-RL's megatron_cfg overwrites fusion flags after construction;
    # finalize() must repair the invalid combination rather than crash later
    p = _tiny_provider()
    p.bias_activation_fusion = True
    p.finalize()
    check(
        "finalize() forces bias_activation_fusion off",
        p.bias_activation_fusion is False,
    )


def test_rope_scaling_applied():
    m0 = _tiny_provider().provide(pre_process=True, post_process=True)
    m1 = _tiny_provider(rope_scaling=True, rope_scaling_factor=32.0).provide(pre_process=True, post_process=True)

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
    from megatron.bridge.models.conversion.model_bridge import (
        ModelConfigNotSupportedError,
    )

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
    check(
        "bridge: model_type registered",
        b.MODEL_TYPE == "apertus",
        f"got {b.MODEL_TYPE!r}",
    )

    try:
        b.hf_config_to_model_config(object())
        check("bridge: generic model config rejected", False, "accepted silently")
    except ModelConfigNotSupportedError as e:
        check("bridge: generic model config rejected", True, str(e)[:60])

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
    check(
        "bridge: rope_scaling=None -> no scaling",
        getattr(p_none, "rope_scaling", "unset") is False,
    )

    # non-default llama3 params must be rejected loudly, not silently dropped
    bad = dict(apertus15, low_freq_factor=2.0)
    try:
        b.provider_bridge(fake_hf(bad))
        check("bridge: non-default rope params rejected", False, "accepted silently")
    except ValueError as e:
        check("bridge: non-default rope params rejected", True, str(e)[:60])

    # transformers v5 shape: rope_theta + scaling live in rope_parameters,
    # and the v4 attributes do not exist at all on the config object
    hf_v5 = fake_hf(None)
    del hf_v5.config.rope_scaling
    del hf_v5.config.rope_theta
    hf_v5.config.rope_parameters = {
        "rope_type": "llama3",
        "rope_theta": 4_000_000,
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    p_v5 = b.provider_bridge(hf_v5)
    check(
        "bridge: transformers-v5 rope_parameters parsed",
        p_v5.rope_scaling is True and p_v5.rope_scaling_factor == 32.0 and p_v5.rotary_base == 4_000_000,
        f"rope_scaling={p_v5.rope_scaling} factor={p_v5.rope_scaling_factor} theta={p_v5.rotary_base}",
    )


def test_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    check(
        "tokenizer chat template present",
        bool(tok.chat_template),
        f"{len(tok.chat_template or '')} chars",
    )


def main():
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}", flush=True)
    test_xielu_module()
    dist_init()
    test_provider_builds_on_stock_mcore()
    test_fusion_invariant_enforced()
    test_rope_scaling_applied()
    test_bridge_parses_full_rope_dict()
    if len(sys.argv) > 1:
        test_tokenizer(sys.argv[1])
    finish()


if __name__ == "__main__":
    main()
