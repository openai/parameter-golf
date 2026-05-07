#!/usr/bin/env bash
# Phase 2b:
#   (a) Refresh xsa_triton.py (now has Triton backward).
#   (b) Re-run numerical + isolated microbench.
#   (c) Block-level bench: FA3 baseline (torch XSA) vs FA3 + Triton XSA, both compiled.
#
# Needs refreshed /workspace/xsa_triton.py uploaded alongside this script.
set -euo pipefail
cd /workspace
cp -f xsa_triton.py work/xsa_triton.py

echo "=== PHASE 2b ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"

# (a) quick sanity: FA3 still importable?
python - <<'PY'
import flash_attn_interface as fa3
print("FA3:", fa3.__file__)
PY

# (b) numerics + isolated bench ---------------------------------------------
echo
echo "--- XSA numerics (fp32 / bf16) ---"
python - <<'PY'
import sys, torch
sys.path.insert(0, "/workspace/work")
from xsa_triton import xsa_triton, xsa_torch

torch.manual_seed(0)
device = torch.device("cuda")
for dtype in (torch.float32, torch.bfloat16):
    print(f"dtype={dtype}")
    for B, T, H, Hkv, D in [(2,128,8,4,64),(8,2048,8,4,64),(4,1024,4,4,64),(2,512,16,4,64)]:
        y  = torch.randn(B,T,H,D,device=device,dtype=dtype,requires_grad=True)
        v  = torch.randn(B,T,Hkv,D,device=device,dtype=dtype,requires_grad=True)
        y2 = y.detach().clone().requires_grad_(); v2 = v.detach().clone().requires_grad_()
        out_t = xsa_triton(y, v); out_r = xsa_torch(y2, v2)
        fwd_err = (out_t - out_r).abs().max().item()
        go = torch.randn_like(out_t)
        out_t.backward(go); out_r.backward(go)
        gy_err = (y.grad - y2.grad).abs().max().item()
        gv_err = (v.grad - v2.grad).abs().max().item()
        print(f"  B={B:<2} T={T:<5} H={H:<2} Hkv={Hkv:<2} D={D}: fwd={fwd_err:.3e} gy={gy_err:.3e} gv={gv_err:.3e}")
PY

echo
echo "--- XSA isolated microbench (B=8 T=2048 H=8 Hkv=4 D=64 bf16) ---"
python - <<'PY'
import sys, time, torch
sys.path.insert(0, "/workspace/work")
from xsa_triton import xsa_triton, xsa_torch

device = torch.device("cuda"); dtype = torch.bfloat16
B,T,H,Hkv,D = 8,2048,8,4,64
y = torch.randn(B,T,H,D,device=device,dtype=dtype,requires_grad=True)
v = torch.randn(B,T,Hkv,D,device=device,dtype=dtype,requires_grad=True)

def make_bench(fn_fwd):
    def run(fwd_only=False):
        y_ = y.detach().clone().requires_grad_()
        v_ = v.detach().clone().requires_grad_()
        out = fn_fwd(y_, v_)
        if not fwd_only:
            out.sum().backward()
    return run

run_eager    = make_bench(xsa_torch)
xsa_torch_c  = torch.compile(xsa_torch, fullgraph=True, dynamic=False)
run_compiled = make_bench(xsa_torch_c)
run_triton   = make_bench(xsa_triton)

def bench(fn, n_warmup=30, n_iter=300):
    for _ in range(n_warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)*1e6/n_iter

print(f"{'impl':<22} {'fwd us':>10} {'fwd+bwd us':>12}")
print("-"*48)
for lbl, fn in [("torch_eager",run_eager),("torch_compiled",run_compiled),("triton",run_triton)]:
    f = bench(lambda: fn(fwd_only=True))
    t = bench(fn)
    print(f"{lbl:<22} {f:>10.1f} {t:>12.1f}")
PY

# (c) Block-level microbench (FA3 baseline vs FA3+Triton XSA) ---------------
echo
echo "--- Block microbench: FA3 baseline vs FA3 + Triton XSA ---"
python - <<'PY'
import os, sys, time, types, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspace/work")
from xsa_triton import xsa_triton as xsa_triton_fn

# Load the FA3-enabled baseline (now that FA3 is importable)
src = open("/workspace/work/train_gpt_baseline.py").read()
ns = {"__name__": "pg_baseline"}
exec(compile(src, "train_gpt_baseline.py", "exec"), ns)
Block, Rotary = ns["Block"], ns["Rotary"]

device = torch.device("cuda"); dtype = torch.bfloat16
torch.manual_seed(0)
B, T, D = 8, 2048, 512
H, KVH = 8, 4

def build_block(use_triton_xsa, layer_idx=7):
    blk = Block(
        dim=D, num_heads=H, num_kv_heads=KVH, mlp_mult=4.0,
        rope_base=10000.0, qk_gain_init=5.0, train_seq_len=T,
        layer_idx=layer_idx, ln_scale=True,
    ).to(device).to(dtype)
    for p in blk.parameters():
        if p.ndim < 2: p.data = p.data.float()
    blk.parallel = True
    blk.attn.use_xsa = True
    blk.attn.rope_dims = 16
    blk.attn.rotary = Rotary(D // H, base=10000.0, train_seq_len=T, rope_dims=16).to(device)
    if use_triton_xsa:
        def _xsa_triton_method(self, y, v): return xsa_triton_fn(y, v)
        blk.attn._xsa_efficient = types.MethodType(_xsa_triton_method, blk.attn)
    return blk

def bench_block(blk_callable, n_warmup=20, n_iter=100):
    x  = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    x0 = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    for _ in range(n_warmup):
        y = blk_callable(x, x0); y.sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        y = blk_callable(x, x0); y.sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0) * 1000.0 / n_iter

print(f"shape: B={B} T={T} D={D} H={H} KVH={KVH}  dtype={dtype}")
print(f"{'variant':<42} {'ms/iter':>10}")
print("-"*56)

for label, use_triton, use_compile in [
    ("eager    FA3  torch-XSA",  False, False),
    ("eager    FA3  Triton-XSA", True,  False),
    ("compiled FA3  torch-XSA",  False, True),
    ("compiled FA3  Triton-XSA", True,  True),
]:
    blk = build_block(use_triton_xsa=use_triton)
    fn = torch.compile(blk, dynamic=False, mode="max-autotune-no-cudagraphs") if use_compile else blk
    try:
        ms = bench_block(fn)
        print(f"  {label:<40} {ms:>10.3f}")
    except Exception as e:
        print(f"  {label:<40}  FAIL: {type(e).__name__}: {str(e)[:80]}")
PY

echo "=== PHASE 2b DONE ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
