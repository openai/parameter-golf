"""
Benchmarks kernel fusion for mamba3 backward to ensure correctness and
performance benefits.

The stock Triton bwd kernels are fully deterministic at our config (model
weights seeded, no atomic_add jitter observed — noise floor is exactly 0),
so --check uses a tight relative-L2 tolerance. Fusion typically introduces
O(bf16 ULP) diffs from reordered fp ops, well below 1e-5.

Usage:
    # Capture a stock reference
    python3 triton_kernels/bench_mamba3_bwd.py --save triton_kernels/ref.pt

    # Run the fused kernel and compare
    MAMBA3_FUSED_BWD=1 python3 triton_kernels/bench_mamba3_bwd.py \
        --check triton_kernels/ref.pt
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import torch

from train_mamba3_hybrid import Hyperparameters, Block

parser = argparse.ArgumentParser()
parser.add_argument("--save", type=str, default=None,
                    help="Save reference grads to this path (typically used with stock kernel)")
parser.add_argument("--check", type=str, default=None,
                    help="Compare computed grads against reference at this path (relative L2)")
parser.add_argument("--l2-tol", type=float, default=1e-5,
                    help="Relative L2 tolerance for --check (||g-r||/||r||). "
                         "Stock kernel is deterministic at our config (noise floor = 0), "
                         "so 1e-5 flags any non-trivial numerical deviation.")
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--iters", type=int, default=50)
cli = parser.parse_args()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b||_2 / ||b||_2 in fp32. Returns inf if b is all zero and a isn't."""
    af = a.float()
    bf = b.float()
    num = (af - bf).norm().item()
    den = bf.norm().item()
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def compare_grad_dicts(got: dict, ref: dict, tol: float, label: str):
    """Relative-L2 compare; returns (n_ok, n_fail, worst_rel, worst_name)."""
    n_ok = n_fail = 0
    worst_rel = 0.0
    worst_name = ""
    for name, g in got.items():
        if name not in ref:
            print(f"  MISSING in ref:  {name}")
            n_fail += 1
            continue
        r = ref[name]
        if g.shape != r.shape:
            print(f"  SHAPE MISMATCH {name}: {tuple(g.shape)} vs {tuple(r.shape)}")
            n_fail += 1
            continue
        rel = rel_l2(g, r)
        if rel > worst_rel:
            worst_rel, worst_name = rel, name
        if rel <= tol:
            n_ok += 1
        else:
            diff = (g.float() - r.float()).abs()
            print(f"  {label} {name}: "
                  f"rel_l2={rel:.2e}  "
                  f"max_abs={diff.max().item():.2e}  "
                  f"shape={tuple(g.shape)}")
            n_fail += 1
    return n_ok, n_fail, worst_rel, worst_name

device = torch.device("cuda")

args = Hyperparameters()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Build a Block then isolate the Mamba3Layer for clean perf attribution.
# Block wraps Mamba3Layer + MLP + norms + residuals; we only want the Mamba3 backward.
# Seed BEFORE construction so model weights are identical across runs — otherwise
# --save runs produce uncorrelated grads (not atomic_add noise) and --check is meaningless.
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
mamba_block = Block(
    args.model_dim, args.mlp_mult,
    args.mamba3_d_state, args.mamba3_expand,
    args.mamba3_headdim, args.mamba3_chunk_size,
    args.mamba3_ngroups, args.mamba3_rope_fraction,
    args.mamba3_outproj_norm,
).to(device).bfloat16()

m3 = mamba_block.mamba3
m3.train()

seq_len = args.train_seq_len
bsz = 131072 // seq_len  # same per-GPU micro-batch as real training

print(f"Model:  {sum(p.numel() for p in m3.parameters())/1e6:.2f}M params (Mamba3Layer only)")
print(f"Batch:  {bsz} seqs x {seq_len} = {bsz * seq_len} tokens")
print(f"Fused:  MAMBA3_FUSED_BWD={os.environ.get('MAMBA3_FUSED_BWD', '0')}")
print()

def zero_grads(inp):
    if inp.grad is not None:
        inp.grad = None
    for p in m3.parameters():
        p.grad = None

# ---- Warmup: covers Triton autotune + compile ----
torch.manual_seed(0)
x_warm = torch.randn(bsz, seq_len, args.model_dim,
                     device=device, dtype=torch.bfloat16, requires_grad=True)
print(f"Warmup ({cli.warmup} iters)...")
for _ in range(cli.warmup):
    zero_grads(x_warm)
    out = m3(x_warm)
    loss = out.float().pow(2).mean()
    loss.backward()
torch.cuda.synchronize()

# ---- Deterministic reference capture ----
# Fresh seeded tensor so --save and --check runs produce identical inputs.
torch.manual_seed(42)
x_ref = torch.randn(bsz, seq_len, args.model_dim,
                    device=device, dtype=torch.bfloat16, requires_grad=True)
zero_grads(x_ref)
out = m3(x_ref)
loss = out.float().pow(2).mean()
loss.backward()

ref_grads = {"__input__": x_ref.grad.detach().clone()}
for name, p in m3.named_parameters():
    if p.grad is not None:
        ref_grads[name] = p.grad.detach().clone()
print(f"Captured {len(ref_grads)} gradient tensors")

if cli.save:
    torch.save(ref_grads, cli.save)
    print(f"Saved reference grads -> {cli.save}")

if cli.check:
    ref = torch.load(cli.check, map_location=device)
    n_ok, n_fail, worst_rel, worst_name = compare_grad_dicts(
        ref_grads, ref, cli.l2_tol, label="MISMATCH")
    print(f"Correctness: {n_ok} OK, {n_fail} FAIL  "
          f"(l2_tol={cli.l2_tol:.2e}, worst={worst_rel:.2e} on {worst_name})")
    if n_fail > 0:
        sys.exit(1)

# ---- Performance measurement ----
print(f"\nBenchmark ({cli.iters} iters)...")
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(cli.iters):
    zero_grads(x_warm)
    out = m3(x_warm)
    loss = out.float().pow(2).mean()
    loss.backward()
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) / cli.iters * 1000
print(f"fwd+bwd: {ms:.2f} ms/iter  (batch={bsz}, seq={seq_len})")
