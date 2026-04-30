#!/usr/bin/env bash
# Phase 1b: FA3->SDPA patch + RMSNorm inspection + Block microbench (eager & compiled).
set -euo pipefail
cd /workspace

echo "=== PHASE 1b: patch FA3 + microbench Block ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"

# 1. Patch the baseline: FA3 import -> SDPA shim ----------------------------
python - <<'PY'
import re, pathlib
src = pathlib.Path("/workspace/work/train_gpt_baseline.py").read_text()

shim = (
    "# --- patched: FA3 -> SDPA shim ---\n"
    "def flash_attn_3_func(q, k, v, causal=False):\n"
    "    import torch.nn.functional as _F\n"
    "    gqa = q.size(-2) != k.size(-2)\n"
    "    y = _F.scaled_dot_product_attention(\n"
    "        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),\n"
    "        is_causal=causal, enable_gqa=gqa,\n"
    "    )\n"
    "    return y.transpose(1, 2)\n"
)

pat = r'from flash_attn_interface import flash_attn_func as flash_attn_3_func\n'
src_new, n = re.subn(pat, shim, src, count=1)
assert n == 1, f"Expected exactly 1 FA3 import, got {n}"
pathlib.Path("/workspace/work/train_gpt_patched.py").write_text(src_new)
print(f"Patched: {len(src)} -> {len(src_new)} bytes, lines: {src_new.count(chr(10))+1}")
PY

# 2. RMSNorm class (confirm parameter-free) ---------------------------------
echo
echo "--- RMSNorm definition ---"
python - <<'PY'
import ast, pathlib
src = pathlib.Path("/workspace/work/train_gpt_patched.py").read_text()
tree = ast.parse(src)
lines = src.splitlines()
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "RMSNorm":
        print("\n".join(lines[node.lineno-1:node.end_lineno]))
        # count nn.Parameter usages inside the class
        n_params = sum(1 for n in ast.walk(node)
                       if isinstance(n, ast.Attribute) and n.attr == "Parameter")
        print(f"\n[RMSNorm has {n_params} nn.Parameter usage(s) in-class]")
PY

# 3. Microbenchmark the parallel-residual block -----------------------------
echo
echo "--- Block microbench (eager & compiled) ---"
python - <<'PY'
import os, time, warnings, torch
warnings.filterwarnings("ignore")

# Exec patched source in a non-main namespace so main() doesn't auto-run
src = open("/workspace/work/train_gpt_patched.py").read()
ns = {"__name__": "pg_patched"}
try:
    exec(compile(src, "train_gpt_patched.py", "exec"), ns)
except SystemExit:
    pass  # just in case main() has a sys.exit
Block = ns["Block"]

device = torch.device("cuda")
dtype = torch.bfloat16
torch.manual_seed(0)

# Bigbag's hyperparams for the hot-path shapes
B, T, D = 8, 2048, 512
H, KVH = 8, 4
MLP_MULT = 4.0

def build_block(parallel, use_xsa, layer_idx=7):
    blk = Block(
        dim=D, num_heads=H, num_kv_heads=KVH, mlp_mult=MLP_MULT,
        rope_base=10000.0, qk_gain_init=5.0, train_seq_len=T,
        layer_idx=layer_idx, ln_scale=True,
    ).to(device).to(dtype)
    # Restore fp32 for the scalar/control params (baseline trick)
    for p in blk.parameters():
        if p.ndim < 2:
            p.data = p.data.float()
    blk.parallel = parallel
    blk.attn.use_xsa = use_xsa
    if hasattr(blk.attn, "rope_dims"):
        # rope_dims=16 per hparams (partial RoPE). Rotary was built with rope_dims=0 in
        # __init__; baseline re-creates it in GPT.__init__ when rope_dims>0. We redo here.
        Rotary = ns["Rotary"]
        blk.attn.rope_dims = 16
        blk.attn.rotary = Rotary(D // H, base=10000.0, train_seq_len=T, rope_dims=16).to(device)
    return blk

def bench(blk_fn, n_warmup=10, n_iter=50, label=""):
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    x0 = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    # warmup (also triggers compile if applicable)
    for _ in range(n_warmup):
        y = blk_fn(x, x0)
        y.sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        y = blk_fn(x, x0)
        y.sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_iter

print(f"shape: B={B} T={T} D={D} H={H} KVH={KVH} MLP_MULT={MLP_MULT} dtype={dtype}")
print(f"{'variant':<48} {'ms/iter':>10}")
print("-" * 60)

for parallel in (False, True):
    for xsa in (False, True):
        blk = build_block(parallel, xsa)
        ms_eager = bench(blk, label=f"eager p={parallel} xsa={xsa}")
        print(f"  eager  parallel={parallel!s:<5} xsa={xsa!s:<5}              {ms_eager:>10.3f}")

# Compiled variant only for the realistic target config
target = build_block(parallel=True, use_xsa=True)
target_compiled = torch.compile(target, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
ms_compiled = bench(target_compiled, n_warmup=20, n_iter=50)
print(f"  compiled parallel=True  xsa=True                  {ms_compiled:>10.3f}")

# 4. Profile the compiled target block
print()
print("--- torch.profiler: compiled parallel+xsa block, 20 fwd+bwd iters ---")
x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
x0 = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
for _ in range(5):
    y = target_compiled(x, x0); y.sum().backward()
torch.cuda.synchronize()
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
    record_shapes=False,
) as prof:
    for _ in range(20):
        y = target_compiled(x, x0); y.sum().backward()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
PY

echo "=== PHASE 1b DONE ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
