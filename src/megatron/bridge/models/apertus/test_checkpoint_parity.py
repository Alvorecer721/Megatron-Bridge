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

"""Parity gate: AutoBridge-built Megatron model vs HF eager, real checkpoint.

Compares logits at short context AND past the rope_scaling original context
(8192), where the llama3 scaling must be active — the regression this gate
exists to catch. Requires 1 GPU (~40GB free) and an HF-format checkpoint.

Run:
    PYTHONPATH=<bridge>/src[:<xielu-site>] python test_checkpoint_parity.py <ckpt_path>
Exit code 0 = parity holds.
"""

import argparse
import gc
import logging

import torch
from _test_harness import check, dist_init, finish


logging.basicConfig(level=logging.INFO)  # surface the XIELU dispatch-path log

SEQ_LENS = [
    128,
    12288,
]  # 12288 > original_max_position_embeddings=8192 -> scaling active
TAIL = 32  # positions compared exactly; full sequence compared by argmax agreement


def make_ids(seq_len, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 100_000, (1, seq_len), generator=g).cuda()


def hf_forward(checkpoint):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    vocab = model.config.vocab_size
    out = {}
    with torch.no_grad():
        for s in SEQ_LENS:
            ids = make_ids(s, seed=s)
            logits = model(input_ids=ids).logits
            top2 = logits[0].topk(2, dim=-1)  # bf16 topk: avoids a full-vocab fp32 temporary
            out[s] = (
                logits[0, -TAIL:].float().cpu(),
                top2.indices[:, 0].cpu(),
                top2.values.float().cpu(),
            )
            del logits, top2
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out, vocab


def megatron_forward(checkpoint, vocab):
    from megatron.bridge import AutoBridge

    dist_init(port=29521)
    bridge = AutoBridge.from_hf_pretrained(checkpoint)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.gradient_accumulation_fusion = False
    if hasattr(provider, "finalize"):
        provider.finalize()
    models = provider.provide_distributed_model(wrap_with_ddp=False)
    model = models[0].cuda().eval()

    out = {}
    with torch.no_grad():
        for s in SEQ_LENS:
            ids = make_ids(s, seed=s)
            pos = torch.arange(s, device="cuda").unsqueeze(0)
            logits = model(input_ids=ids, position_ids=pos, attention_mask=None)
            if logits.shape[0] != 1:  # [s, b, v] -> [b, s, v]
                logits = logits.transpose(0, 1)
            logits = logits[..., :vocab]  # drop padded-vocab columns
            out[s] = (logits[0, -TAIL:].float().cpu(), logits[0].argmax(-1).cpu())
            del logits
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="HF-format Apertus checkpoint path")
    args = parser.parse_args()

    print(f"checkpoint: {args.checkpoint}", flush=True)
    hf, vocab = hf_forward(args.checkpoint)
    print("HF forward done", flush=True)
    mg = megatron_forward(args.checkpoint, vocab)
    print("Megatron forward done", flush=True)

    for s in SEQ_LENS:
        hf_tail, hf_arg, hf_top2 = hf[s]
        mg_tail, mg_arg = mg[s]
        max_diff = (hf_tail - mg_tail).abs().max().item()
        mean_diff = (hf_tail - mg_tail).abs().mean().item()
        agree_all = (hf_arg == mg_arg).float().mean().item()
        # argmax flips inside bf16 noise are expected on near-tie positions;
        # only positions where HF is confident (top1-top2 margin) are meaningful
        confident = (hf_top2[:, 0] - hf_top2[:, 1]) > 0.5
        agree_conf = (hf_arg[confident] == mg_arg[confident]).float().mean().item()
        check(
            f"seq={s}: confident top-1 agreement >= 0.995",
            agree_conf >= 0.995,
            f"confident={agree_conf:.4f} (n={int(confident.sum())}) overall={agree_all:.4f}",
        )
        check(
            f"seq={s}: tail logits close",
            mean_diff < 0.25 and max_diff < 2.0,
            f"mean_abs={mean_diff:.4f} max_abs={max_diff:.4f}",
        )

    finish()


if __name__ == "__main__":
    main()
