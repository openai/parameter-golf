#!/usr/bin/env bash
# Phase 2a:
#   1. Try to install FA3 wheel (cu128, various torch versions).
#   2. Verify the XSA Triton kernel: numerical correctness fwd+bwd vs torch reference.
#   3. Microbench XSA kernel vs torch.compile'd reference.
#
# Needs /workspace/xsa_triton.py uploaded alongside this script.
set -euo pipefail
cd /workspace
mkdir -p work
cp -f xsa_triton.py work/xsa_triton.py

echo "=== PHASE 2a ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"

# 1. FA3 install attempts ---------------------------------------------------
echo
echo "--- FA3 install attempts ---"
FA3_OK=0
for TORCH_TAG in cu128_torch280 cu128_torch281 cu128_torch290 cu128_torch291; do
    URL="https://windreamer.github.io/flash-attention3-wheels/${TORCH_TAG}/"
    echo "  trying $URL"
    if pip install --quiet --no-deps flash_attn_3 --find-links "$URL" 2>&1 | tail -3; then
        if python -c "import flash_attn_interface" 2>/dev/null; then
            echo "  FA3 installed OK from $TORCH_TAG"
            FA3_OK=1
            break
        else
            pip uninstall -y flash_attn_3 2>/dev/null || true
        fi
    fi
done
if [ "$FA3_OK" = "0" ]; then
    echo "  FA3 install FAILED; continuing with SDPA fallback"
fi

python - <<'PY'
try:
    import flash_attn_interface as fa3
    print("FA3 available:", getattr(fa3, "__version__", "?"), fa3.__file__)
except Exception as e:
    print("FA3 NOT available:", type(e).__name__, e)
PY

# 2. Numerical correctness --------------------------------------------------
echo
echo "--- XSA numerical check (fwd + bwd) ---"
python - <<'PY'
import sys, torch
sys.path.insert(0, "/workspace/work")
from xsa_triton import xsa_triton, xsa_torch

torch.manual_seed(0)
device = torch.device("cuda")

# Multiple shape configs
configs = [
    # (B, T, H, Hkv, D)
    (2, 128, 8, 4, 64),     # tiny
    (8, 2048, 8, 4, 64),    # realistic (bigbag)
    (4, 1024, 4, 4, 64),    # no GQA (group=1)
    (2, 512, 16, 4, 64),    # group=4
]

for dtype in (torch.float32, torch.bfloat16):
    print(f"\ndtype = {dtype}")
    print(f"{'shape':<30} {'fwd_max':>12} {'gy_max':>12} {'gv_max':>12}")
    for B, T, H, Hkv, D in configs:
        y = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, T, Hkv, D, device=device, dtype=dtype, requires_grad=True)
        y2 = y.detach().clone().requires_grad_()
        v2 = v.detach().clone().requires_grad_()

        out_t = xsa_triton(y, v)
        out_r = xsa_torch(y2, v2)
        fwd_err = (out_t - out_r).abs().max().item()

        grad_out = torch.randn_like(out_t)
        out_t.backward(grad_out)
        out_r.backward(grad_out)
        gy_err = (y.grad - y2.grad).abs().max().item()
        gv_err = (v.grad - v2.grad).abs().max().item()
        print(f"  B={B:<2} T={T:<5} H={H:<2} Hkv={Hkv:<2} D={D:<3}  {fwd_err:>12.3e} {gy_err:>12.3e} {gv_err:>12.3e}")
PY

# 3. Isolated microbench ----------------------------------------------------
echo
echo "--- XSA microbench (isolated op, B=8 T=2048 H=8 Hkv=4 D=64, bf16) ---"
python - <<'PY'
import sys, time, torch
sys.path.insert(0, "/workspace/work")
from xsa_triton import xsa_triton, xsa_torch

device = torch.device("cuda")
dtype = torch.bfloat16
B, T, H, Hkv, D = 8, 2048, 8, 4, 64

y = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(B, T, Hkv, D, device=device, dtype=dtype, requires_grad=True)

# torch reference (eager)
def run_torch(fwd_only=False):
    y_ = y.detach().clone().requires_grad_()
    v_ = v.detach().clone().requires_grad_()
    out = xsa_torch(y_, v_)
    if fwd_only:
        return
    out.sum().backward()

# torch reference (compiled)
xsa_torch_c = torch.compile(xsa_torch, fullgraph=True, dynamic=False)
def run_torch_compiled(fwd_only=False):
    y_ = y.detach().clone().requires_grad_()
    v_ = v.detach().clone().requires_grad_()
    out = xsa_torch_c(y_, v_)
    if fwd_only:
        return
    out.sum().backward()

def run_triton(fwd_only=False):
    y_ = y.detach().clone().requires_grad_()
    v_ = v.detach().clone().requires_grad_()
    out = xsa_triton(y_, v_)
    if fwd_only:
        return
    out.sum().backward()

def bench(fn, n_warmup=20, n_iter=200, label=""):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / n_iter   # us per call

print(f"{'impl':<24} {'fwd us':>10} {'fwd+bwd us':>12}")
print("-" * 50)
for label, fn in [("torch_eager",    run_torch),
                  ("torch_compiled", run_torch_compiled),
                  ("triton",         run_triton)]:
    us_fwd = bench(lambda: fn(fwd_only=True), label=label + "_fwd")
    us_full = bench(fn, label=label + "_full")
    print(f"{label:<24} {us_fwd:>10.1f} {us_full:>12.1f}")
PY

echo "=== PHASE 2a DONE ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
