"""Profile compiled model on 1-GPU with chrome trace."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from torch.profiler import profile, ProfilerActivity

# Set env vars to match our standard config
for k, v in {
    "FP16_INPROJ_ROWS": "0", "WARMDOWN_ITERS": "2600", "WARMDOWN_SHAPE": "linear",
    "MUON_EQ_R": "1", "LATE_QAT_THRESHOLD": "0.15", "WEIGHT_DECAY": "0.04",
    "MUON_MOMENTUM": "0.99", "MATRIX_LR": "0.025", "EVAL_STRIDE": "32",
}.items():
    os.environ.setdefault(k, v)

from train_mamba3_hybrid import Hyperparameters, GPT

args = Hyperparameters()
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Build model
model = GPT(
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

print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

seq_len = args.train_seq_len
bsz = 131072 // seq_len
x = torch.randint(0, args.vocab_size, (bsz, seq_len), device=device)
y = torch.randint(0, args.vocab_size, (bsz, seq_len), device=device)
print(f"Batch: {bsz} seqs x {seq_len} tokens = {bsz * seq_len} tokens/step")

# Compile
compiled = torch.compile(model, dynamic=False, fullgraph=False)

# Warmup (covers Triton autotune + torch.compile JIT)
print("Warming up (30 steps)...")
compiled.train()
for i in range(30):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = compiled(x, y)
    loss.backward()
    compiled.zero_grad()
    if i % 10 == 0:
        print(f"  warmup step {i}")
torch.cuda.synchronize()

# Benchmark
print("Benchmarking (50 steps)...")
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = compiled(x, y)
    loss.backward()
    compiled.zero_grad()
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / 50 * 1000
print(f"Compiled: {ms:.1f} ms/step")

# Profile with trace
print("Profiling (5 steps)...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(5):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = compiled(x, y)
        loss.backward()
        compiled.zero_grad()
        torch.cuda.synchronize()

prof.export_chrome_trace("profiling/trace_compiled.json")
print(f"Trace saved: profiling/trace_compiled.json")

print("\nTOP 20 SELF CUDA TIME")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
