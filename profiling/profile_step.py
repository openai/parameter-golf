"""Profile a few training steps to understand where wall time goes.
Outputs a chrome trace (viewable in chrome://tracing or Perfetto)
and a summary table of top kernels."""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule

# Set env vars to match our standard config
for k, v in {
    "FP16_INPROJ_ROWS": "0", "WARMDOWN_ITERS": "2600", "WARMDOWN_SHAPE": "linear",
    "MUON_EQ_R": "1", "LATE_QAT_THRESHOLD": "0.15", "WEIGHT_DECAY": "0.04",
    "MUON_MOMENTUM": "0.99", "MATRIX_LR": "0.025", "EVAL_STRIDE": "32",
}.items():
    os.environ.setdefault(k, v)

from train_mamba3_hybrid import Hyperparameters, GPT, load_validation_tokens

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

# Fake batch (same shape as training)
seq_len = args.train_seq_len
# micro_batch = train_batch_tokens / 8 = 1M / 8 = 131072 tokens = 32 seqs @ 4096
bsz = 131072 // seq_len
x = torch.randint(0, args.vocab_size, (bsz, seq_len), device=device)
y = torch.randint(0, args.vocab_size, (bsz, seq_len), device=device)

print(f"Batch: {bsz} seqs x {seq_len} tokens = {bsz * seq_len} tokens/step")

# Warmup (covers Triton autotune)
print("Warming up (20 steps)...")
model.train()
for i in range(20):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    loss.backward()
    model.zero_grad()
    if i % 5 == 0:
        print(f"  warmup step {i}")

torch.cuda.synchronize()
print("Warmup done. Profiling...")

# Profile 5 steps
NUM_PROFILE_STEPS = 5

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    for step in range(NUM_PROFILE_STEPS):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        model.zero_grad()
        torch.cuda.synchronize()

# Export chrome trace
trace_path = "profiling/trace_mamba3_hybrid.json"
prof.export_chrome_trace(trace_path)
print(f"\nChrome trace saved to: {trace_path}")
print("Open in chrome://tracing or https://ui.perfetto.dev/\n")

# Print summary tables
print("=" * 80)
print("TOP 30 CUDA KERNELS BY TOTAL TIME")
print("=" * 80)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

print("\n" + "=" * 80)
print("TOP 20 BY SELF CUDA TIME")
print("=" * 80)
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

# Count kernel launches
events = prof.key_averages()
total_cuda_time = sum(e.cuda_time_total for e in events if e.cuda_time_total > 0)
total_calls = sum(e.count for e in events if e.cuda_time_total > 0)
print(f"\nTotal CUDA time: {total_cuda_time / 1e6:.1f}s over {NUM_PROFILE_STEPS} steps")
print(f"Total kernel calls: {total_calls} ({total_calls // NUM_PROFILE_STEPS}/step)")
print(f"Avg step time: {total_cuda_time / NUM_PROFILE_STEPS / 1e6:.3f}s")
