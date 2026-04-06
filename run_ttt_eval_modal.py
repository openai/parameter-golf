"""TTT-only eval on the Exp 2 quantized model. No training — just eval.

Downloads exp2_w256_full5_seed1337.int6.ptz from HuggingFace,
dequantizes, loads into eval model with matching SWA config,
runs sliding window eval + Legal Score-First TTT.
"""
import os
import modal

app = modal.App("parameter-golf-ttt-eval")

data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token:
    _hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(_hf_token_path):
        _hf_token = open(_hf_token_path).read().strip()
hf_secret = modal.Secret.from_dict({"HF_TOKEN": _hf_token})

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "torch==2.9.1",
        "numpy",
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "tqdm",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install("psutil", "packaging", "ninja", "wheel", "setuptools")
    .run_commands(
        "pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291",
    )
    .add_local_file("train_gpt_swa.py", "/opt/train_gpt_swa.py")
)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=5400,
    volumes={"/data": data_vol},
    secrets=[hf_secret],
)
def ttt_eval():
    import subprocess
    import os
    import shutil
    import io
    import lzma
    import time
    import torch

    os.makedirs("/workspace/parameter-golf", exist_ok=True)
    os.chdir("/workspace/parameter-golf")
    shutil.copy2("/opt/train_gpt_swa.py", "train_gpt_swa.py")

    # Dataset
    dataset_vol = "/data/fineweb10B_sp1024"
    dataset_local = "./data/datasets/fineweb10B_sp1024"
    tokenizer_vol = "/data/tokenizers"
    tokenizer_local = "./data/tokenizers"

    if not os.path.exists(f"{dataset_vol}/fineweb_train_000000.bin"):
        print("Downloading dataset...", flush=True)
        subprocess.run(
            ["git", "clone", "https://github.com/Itssshikhar/parameter-golf.git", "/tmp/repo"],
            check=True,
        )
        shutil.copytree("/tmp/repo/data", "./data", dirs_exist_ok=True)
        subprocess.run(
            ["python3", "data/cached_challenge_fineweb.py", "--variant", "sp1024"],
            check=True,
        )
        os.makedirs(dataset_vol, exist_ok=True)
        os.makedirs(tokenizer_vol, exist_ok=True)
        for f in os.listdir(dataset_local):
            shutil.copy2(f"{dataset_local}/{f}", f"{dataset_vol}/{f}")
        for f in os.listdir(tokenizer_local):
            shutil.copy2(f"{tokenizer_local}/{f}", f"{tokenizer_vol}/{f}")
        data_vol.commit()
    else:
        print("Dataset found in volume.", flush=True)
        os.makedirs("./data/datasets", exist_ok=True)
        os.makedirs("./data", exist_ok=True)
        if os.path.exists(dataset_local):
            shutil.rmtree(dataset_local)
        os.symlink(dataset_vol, dataset_local)
        if os.path.exists(tokenizer_local):
            shutil.rmtree(tokenizer_local)
        os.symlink(tokenizer_vol, tokenizer_local)

    # Download the quantized model from HuggingFace
    print("Downloading quantized model from HuggingFace...", flush=True)
    from huggingface_hub import hf_hub_download
    ptz_path = hf_hub_download(
        "shikhar007/parameter-golf-gram-ns",
        "exp2_w256_full5_seed1337.int6.ptz",
    )
    print(f"Downloaded: {ptz_path}", flush=True)

    # Copy the TTT function from the PR #549 codebase (train_gpt_gram_ns.py)
    # into the eval script since train_gpt_swa.py doesn't have it.
    eval_script = f'''
import io, lzma, os, sys, time, math, glob, torch, numpy as np, sentencepiece as spm
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from pathlib import Path

sys.path.insert(0, ".")
os.environ["SWA_WINDOW_SIZE"] = "256"
os.environ["SWA_FULL_ATTN_LAYERS"] = "5"
os.environ["BIGRAM_VOCAB_SIZE"] = "3072"
os.environ["BIGRAM_DIM"] = "112"
os.environ["NUM_LAYERS"] = "11"
os.environ["XSA_LAST_N"] = "11"
os.environ["ROPE_DIMS"] = "16"
os.environ["LN_SCALE"] = "1"
os.environ["VE_ENABLED"] = "1"
os.environ["VE_LAYERS"] = "9,10"
os.environ["TTT_ENABLED"] = "1"
os.environ["TTT_EPOCHS"] = "3"
os.environ["TTT_FREEZE_BLOCKS"] = "0"
os.environ["TTT_LR"] = "0.002"
os.environ["TTT_CHUNK_TOKENS"] = "32768"
os.environ["TTT_MOMENTUM"] = "0.9"
os.environ["TTT_BATCH_SEQS"] = "32"
os.environ["TTT_GRAD_CLIP"] = "1.0"

import train_gpt_swa as tgs

distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
if distributed:
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
master = rank == 0
grad_accum_steps = 8 // world_size

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

def log0(msg):
    if master: print(msg, flush=True)

args = tgs.Hyperparameters()
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
val_tokens = tgs.load_validation_tokens(args.val_files, args.train_seq_len)
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tgs.build_sentencepiece_luts(sp, args.vocab_size, device)
effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len

# Load quantized model
log0("Loading quantized model...")
with open("{ptz_path}", "rb") as f:
    quant_blob = f.read()
quant_obj = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")

# Build template state dict for dequantization
model_kwargs = dict(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    gated_attention=args.gated_attention, value_residual=args.value_residual,
)
template_model = tgs.GPT(**model_kwargs)
template_sd = {{k: v.detach().cpu() for k, v in template_model.state_dict().items()}}
unbanked_template = tgs._unbank_state_dict(template_sd, args.num_layers)
deq_unbanked = tgs.dequantize_mixed_int6(quant_obj["w"], quant_obj["m"], unbanked_template)
deq_state = tgs._rebank_state_dict(deq_unbanked, args.num_layers, template_sd)

# Load into eval model
eval_model = tgs.GPT(**model_kwargs).to(device).bfloat16()
eval_model.qo_bank.data = eval_model.qo_bank.data.float()
eval_model.kv_bank.data = eval_model.kv_bank.data.float()
eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
for m in eval_model.modules():
    if isinstance(m, tgs.CastedLinear):
        m.float()
tgs.restore_low_dim_params_to_fp32(eval_model)
eval_model.load_state_dict(deq_state, strict=True)

log0(f"SWA layers: {{eval_model._swa_layers}}")
log0(f"Window sizes: {{[b.attn.window_size for b in eval_model.blocks]}}")
log0(f"Model params: {{sum(p.numel() for p in eval_model.parameters()):,}}")
del template_model, template_sd, unbanked_template, deq_unbanked, deq_state

# Standard roundtrip eval
log0("Running standard roundtrip eval...")
compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
torch.cuda.synchronize(); t0 = time.perf_counter()
rt_loss, rt_bpb = tgs.eval_val(
    args, compiled_eval, rank, world_size, device, grad_accum_steps,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    eval_seq_len=effective_eval_seq_len,
)
torch.cuda.synchronize()
log0(f"roundtrip val_loss:{{rt_loss:.4f}} val_bpb:{{rt_bpb:.4f}} time:{{1000*(time.perf_counter()-t0):.0f}}ms")

# Sliding window eval
log0("Running sliding window eval...")
torch.cuda.synchronize(); t0 = time.perf_counter()
sw_loss, sw_bpb = tgs.eval_val_sliding(
    args, eval_model, rank, world_size, device,
    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride=64, eval_seq_len=effective_eval_seq_len,
)
torch.cuda.synchronize()
log0(f"sliding val_loss:{{sw_loss:.4f}} val_bpb:{{sw_bpb:.4f}} time:{{1000*(time.perf_counter()-t0):.0f}}ms")

# --- TTT eval (inlined from train_gpt_gram_ns.py) ---
log0("Running TTT eval (3 epochs, all blocks, lr=0.002)...")

ttt_lr = 0.002
ttt_epochs = 3
ttt_chunk_tokens = 32768
ttt_freeze_blocks = 0
ttt_momentum = 0.9
ttt_batch_seqs = 32
ttt_grad_clip = 1.0
seq_len = args.train_seq_len
total_tokens = val_tokens.numel() - 1
stride = 64

window_starts = [ws for ws in range(0, total_tokens, stride)
                 if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
num_chunks = (total_tokens + ttt_chunk_tokens - 1) // ttt_chunk_tokens
chunk_windows = [[] for _ in range(num_chunks)]
for ws in window_starts:
    end = min(ws + seq_len, total_tokens)
    wlen = end - ws
    s = 0 if ws == 0 else max(wlen - stride, 0)
    scored_start = ws + s
    ci = min(scored_start // ttt_chunk_tokens, num_chunks - 1)
    chunk_windows[ci].append(ws)

log0(f"ttt:start chunks={{num_chunks}} windows={{len(window_starts)}}")

loss_sum = torch.zeros((), device=device, dtype=torch.float64)
token_count = torch.zeros((), device=device, dtype=torch.float64)
byte_count = torch.zeros((), device=device, dtype=torch.float64)

ttt_params = [p for p in eval_model.parameters()]
for p in ttt_params:
    p.requires_grad_(True)
optimizer = torch.optim.SGD(ttt_params, lr=ttt_lr, momentum=ttt_momentum)

torch.cuda.synchronize(); t0 = time.perf_counter()

for ci in range(num_chunks):
    windows = chunk_windows[ci]
    if not windows:
        continue

    chunk_start = ci * ttt_chunk_tokens
    chunk_end = min((ci + 1) * ttt_chunk_tokens, total_tokens)

    # Phase 1: SCORE (inference_mode)
    my_s = (len(windows) * rank) // world_size
    my_e = (len(windows) * (rank + 1)) // world_size
    my_windows = windows[my_s:my_e]

    eval_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), ttt_batch_seqs):
            batch_ws = my_windows[bi:bi + ttt_batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk_tok[:-1]
                y_batch[i, :wlen] = chunk_tok[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = eval_model.forward_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1), reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    # Phase 2: TRAIN on scored chunk
    is_last_chunk = (ci == num_chunks - 1)
    if not is_last_chunk and ttt_epochs > 0:
        eval_model.train()
        chunk_seqs = (chunk_end - chunk_start) // seq_len
        if chunk_seqs > 0:
            cos_lr = ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
            for pg in optimizer.param_groups:
                pg["lr"] = cos_lr
            my_seq_s = (chunk_seqs * rank) // world_size
            my_seq_e = (chunk_seqs * (rank + 1)) // world_size
            my_chunk_seqs = my_seq_e - my_seq_s
            for _ep in range(ttt_epochs):
                for bs in range(0, my_chunk_seqs, ttt_batch_seqs):
                    be = min(bs + ttt_batch_seqs, my_chunk_seqs)
                    actual_bs = my_seq_s + bs
                    start_tok = chunk_start + actual_bs * seq_len
                    end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                    if end_tok > val_tokens.numel():
                        continue
                    local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                    x = local[:-1].reshape(-1, seq_len)
                    y = local[1:].reshape(-1, seq_len)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = eval_model(x, y)
                    loss.backward()
                    if world_size > 1:
                        for p in ttt_params:
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    torch.nn.utils.clip_grad_norm_(ttt_params, ttt_grad_clip)
                    optimizer.step()

    if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
        elapsed = time.perf_counter() - t0
        rl = loss_sum.item() / max(token_count.item(), 1)
        rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
        log0(f"  ttt_chunk [{{ci+1}}/{{num_chunks}}] bpb={{rbpb:.6f}} time={{elapsed:.1f}}s")

if dist.is_available() and dist.is_initialized():
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

ttt_loss = (loss_sum / token_count).item()
ttt_bpb = ttt_loss / math.log(2.0) * (token_count.item() / byte_count.item())
torch.cuda.synchronize()
log0(f"ttt val_loss:{{ttt_loss:.4f}} val_bpb:{{ttt_bpb:.4f}} time:{{1000*(time.perf_counter()-t0):.0f}}ms")

log0(f"SUMMARY: roundtrip={{rt_bpb:.4f}} sliding={{sw_bpb:.4f}} ttt={{ttt_bpb:.4f}}")

if distributed:
    dist.destroy_process_group()
'''

    with open("eval_ttt.py", "w") as f:
        f.write(eval_script)

    env = {
        **os.environ,
        "SWA_WINDOW_SIZE": "256",
        "SWA_FULL_ATTN_LAYERS": "5",
        "BIGRAM_VOCAB_SIZE": "3072",
        "BIGRAM_DIM": "112",
        "TTT_ENABLED": "1",
        "TTT_EPOCHS": "3",
        "TTT_FREEZE_BLOCKS": "0",
        "TTT_LR": "0.002",
        "TTT_CHUNK_TOKENS": "32768",
    }

    print("=== TTT EVAL START ===", flush=True)
    proc = subprocess.Popen(
        ["torchrun", "--standalone", "--nproc_per_node=8", "eval_ttt.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    for line in proc.stdout:
        line = line.rstrip()
        print(line, flush=True)
        lines.append(line)

    proc.wait()
    print(f"=== TTT EVAL DONE (exit code {proc.returncode}) ===", flush=True)
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    log = ttt_eval.remote()
    with open("ttt_eval.log", "w") as f:
        f.write(log)
    print(f"\nLog saved to ttt_eval.log")
