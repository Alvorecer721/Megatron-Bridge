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

"""Phase-2 parity gate: AutoBridge-built Megatron model vs HF eager, real checkpoint.

Compares logits at short context AND past the rope_scaling original context
(8192), where the llama3 factor-32 scaling must be active — the regression
this gate exists to catch. Requires 1 GPU (~40GB free) and the checkpoint.

Run:
    PYTHONPATH=<deps>:<bridge>/src:<xielu-site> python test_checkpoint_parity.py [ckpt_path]
Exit code 0 = parity holds.
"""

import gc
import logging
import os
import sys

import torch

logging.basicConfig(level=logging.INFO)  # surface the XIELU dispatch-path log

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "/capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus-1p5-8B-sft-16k-lr6e-5-constant-it38036"
)
SEQ_LENS = [128, 12288]  # 12288 > original_max_position_embeddings=8192 -> scaling active
TAIL = 32  # positions compared exactly; full sequence compared by argmax agreement

failures = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    if not cond:
        failures.append(name)


def make_ids(seq_len, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 100_000, (1, seq_len), generator=g).cuda()


def hf_forward():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16).cuda().eval()
    out = {}
    with torch.no_grad():
        for s in SEQ_LENS:
            ids = make_ids(s, seed=s)
            logits = model(input_ids=ids).logits
            top2 = logits[0].float().topk(2, dim=-1)
            out[s] = (logits[0, -TAIL:].float().cpu(), top2.indices[:, 0].cpu(), top2.values.cpu())
            del logits, top2
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def megatron_forward():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from megatron.bridge import AutoBridge

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29521")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl", world_size=1, rank=0)
    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(42)

    bridge = AutoBridge.from_hf_pretrained(CKPT)
    models = bridge.to_megatron_model(load_weights=True, wrap_with_ddp=False)
    model = models[0].cuda().eval()

    out = {}
    with torch.no_grad():
        for s in SEQ_LENS:
            ids = make_ids(s, seed=s)
            pos = torch.arange(s, device="cuda").unsqueeze(0)
            logits = model(input_ids=ids, position_ids=pos, attention_mask=None)
            if logits.shape[0] != 1:  # [s, b, v] -> [b, s, v]
                logits = logits.transpose(0, 1)
            logits = logits[..., :266752]  # drop any vocab padding
            out[s] = (logits[0, -TAIL:].float().cpu(), logits[0].argmax(-1).cpu())
            del logits
    return out


def main():
    print(f"checkpoint: {CKPT}", flush=True)
    hf = hf_forward()
    print("HF forward done", flush=True)
    mg = megatron_forward()
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

    print(f"\n{len(failures)} failure(s)" if failures else "\nCHECKPOINT PARITY GATE PASSED", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
