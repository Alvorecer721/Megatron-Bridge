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

"""Converter audit: raw Megatron-LM dist-ckpt vs HF conversion, same skeleton.

Builds the Apertus model twice from the same HF config via the bridge —
once with weights converted from HF safetensors, once with weights loaded
directly from the raw Megatron-LM torch_dist checkpoint — and compares
logits on identical inputs. Bit-equal weights must give (near-)bit-equal
logits; any conversion drift shows up directly.

Run (1 GPU, ~40GB):
    PYTHONPATH=<bridge>/src python test_raw_ckpt_parity.py <hf_ckpt> <raw_iter_dir>
"""

import sys

import torch

from _test_harness import check, dist_init, finish

HF_CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200"
)
RAW_ITER = sys.argv[2] if len(sys.argv) > 2 else (
    "/capstor/store/cscs/swissai/infra01/apertus_1p5/Megatron-LM-8B/logs/Meg-Runs/main-runs-v2-apertus-1p5"
    "/long_context_sft/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n/checkpoints/iter_0004200"
)
SEQ_LENS = [128, 12288]


def make_ids(seq_len, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 100_000, (1, seq_len), generator=g).cuda()


def forward_logits(model, vocab):
    out = {}
    with torch.no_grad():
        for s in SEQ_LENS:
            ids = make_ids(s, seed=s)
            pos = torch.arange(s, device="cuda").unsqueeze(0)
            logits = model(input_ids=ids, position_ids=pos, attention_mask=None)
            if logits.shape[0] != 1:
                logits = logits.transpose(0, 1)
            out[s] = logits[0, :, :vocab].float().cpu()
            del logits
    return out


def main():
    from megatron.core import dist_checkpointing

    from megatron.bridge import AutoBridge

    dist_init(port=29531)
    bridge = AutoBridge.from_hf_pretrained(HF_CKPT)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.gradient_accumulation_fusion = False  # no apex in this env
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)[0].cuda().eval()
    vocab = 266752
    hf_logits = forward_logits(model, vocab)
    print("HF-converted forward done", flush=True)

    # overwrite the same skeleton's weights from the raw torch_dist checkpoint
    sharded = model.sharded_state_dict(prefix="")
    loaded = dist_checkpointing.load(sharded, RAW_ITER)
    # the load also surfaces the checkpoint's common state (args, optimizer,
    # counters, ...) — keep only what the model actually asked for
    loaded = {k: v for k, v in loaded.items() if k in sharded}
    # dist_checkpointing.load raises if any requested tensor is absent from
    # the checkpoint; spot-check that the apertus-specific params came along.
    # (MegatronModule.load_state_dict returns None, unlike stock nn.Module.)
    model.load_state_dict(loaded)
    flat = list(loaded)
    check(
        "raw load: alpha params present",
        any("alpha_p" in k for k in flat) and any("q_layernorm" in k for k in flat),
        f"{len(flat)} top-level entries",
    )
    raw_logits = forward_logits(model, vocab)
    print("raw-ckpt forward done", flush=True)

    for s in SEQ_LENS:
        d = (hf_logits[s] - raw_logits[s]).abs()
        check(
            f"seq={s}: HF-converted == raw-ckpt logits",
            d.max().item() < 1e-3,
            f"max_abs={d.max().item():.3e} mean_abs={d.mean().item():.3e}",
        )

    finish()


if __name__ == "__main__":
    main()
