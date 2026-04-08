"""Profile 8-GPU DDP training: baseline vs compiled.
Run with: torchrun --nproc_per_node=8 profiling/profile_ddp.py"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time

for k, v in {
    "FP16_INPROJ_ROWS": "0", "WARMDOWN_ITERS": "2600", "WARMDOWN_SHAPE": "linear",
    "MUON_EQ_R": "1", "LATE_QAT_THRESHOLD": "0.15", "WEIGHT_DECAY": "0.04",
    "MUON_MOMENTUM": "0.99", "MATRIX_LR": "0.025", "EVAL_STRIDE": "32",
}.items():
    os.environ.setdefault(k, v)

from train_mamba3_hybrid import Hyperparameters, GPT, CastedLinear, restore_low_dim_params_to_fp32

args = Hyperparameters()
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def log0(msg):
    if rank == 0:
        print(msg, flush=True)

# Build model (same as training script)
base_model = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap,
    use_smeargate=args.use_smeargate, use_bigram_hash=args.use_bigram_hash,
    bigram_buckets=args.bigram_buckets, bigram_hash_dim=args.bigram_hash_dim,
    use_ortho_init=args.use_ortho_init,
    mamba3_d_state=args.mamba3_d_state, mamba3_expand=args.mamba3_expand,
    mamba3_headdim=args.mamba3_headdim, mamba3_chunk_size=args.mamba3_chunk_size,
    mamba3_ngroups=args.mamba3_ngroups, mamba3_rope_fraction=args.mamba3_rope_fraction,
    mamba3_outproj_norm=args.mamba3_outproj_norm,
    num_attn_layers=args.num_attn_layers, num_heads=args.num_heads,
    num_kv_heads=args.num_kv_heads, rope_base=args.rope_base,
    qk_gain_init=args.qk_gain_init,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim,
).to(device).bfloat16()

for module in base_model.modules():
    if isinstance(module, CastedLinear):
        module.float()
restore_low_dim_params_to_fp32(base_model)

log0(f"Model: {sum(p.numel() for p in base_model.parameters())/1e6:.1f}M params, {world_size} GPUs")

# Fake batch — same micro-batch per GPU as real training
seq_len = args.train_seq_len
bsz = 131072 // seq_len  # same as train: 1M / 8 GPUs = 131072 tokens/GPU
x = torch.randint(0, args.vocab_size, (bsz, seq_len), device=device)
y = torch.randint(0, args.vocab_size, (bsz, seq_len), device=device)
log0(f"Batch per GPU: {bsz} seqs x {seq_len} = {bsz * seq_len} tokens")


def bench(model, label, warmup=20, steps=50):
    model.train()
    for _ in range(warmup):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    dist.barrier()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = elapsed / steps * 1000
    log0(f"[{label}] {ms:.1f} ms/step ({steps} steps)")
    return ms


# --- 1. DDP compiled (baseline) ---
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
model_compiled = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
ms_baseline = bench(model_compiled, "compiled+DDP", warmup=25)
del model_compiled

# --- 2. DDP compiled + static_graph ---
compiled_model2 = torch.compile(base_model, dynamic=False, fullgraph=False)
model_static = DDP(compiled_model2, device_ids=[local_rank], broadcast_buffers=False, static_graph=True)
ms_static = bench(model_static, "compiled+DDP+static_graph", warmup=25)
del model_static

# --- 3. DDP compiled + small buckets (5MB) ---
compiled_model3 = torch.compile(base_model, dynamic=False, fullgraph=False)
model_small_bucket = DDP(compiled_model3, device_ids=[local_rank], broadcast_buffers=False,
                         bucket_cap_mb=5)
ms_small_bucket = bench(model_small_bucket, "compiled+DDP+5MB_buckets", warmup=25)
del model_small_bucket

# --- 4. DDP compiled + gradient_as_bucket_view ---
compiled_model4 = torch.compile(base_model, dynamic=False, fullgraph=False)
model_bucket_view = DDP(compiled_model4, device_ids=[local_rank], broadcast_buffers=False,
                        gradient_as_bucket_view=True)
ms_bucket_view = bench(model_bucket_view, "compiled+DDP+grad_bucket_view", warmup=25)
del model_bucket_view

log0(f"\n{'='*50}")
log0(f"compiled+DDP (baseline):     {ms_baseline:.1f} ms/step")
log0(f"+ static_graph:              {ms_static:.1f} ms/step")
log0(f"+ 5MB buckets:               {ms_small_bucket:.1f} ms/step")
log0(f"+ gradient_as_bucket_view:   {ms_bucket_view:.1f} ms/step")

dist.destroy_process_group()
