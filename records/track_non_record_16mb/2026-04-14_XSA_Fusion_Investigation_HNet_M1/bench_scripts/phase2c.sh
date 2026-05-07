#!/usr/bin/env bash
# Phase 2c:
#   (a) Inspect FA3 module + standalone FA3-vs-SDPA attention bench.
#   (b) Add QKV fusion (stack c_q/c_k/c_v weights into single GEMM) via a helper.
#   (c) Block-level bench across the full cross product:
#         {eager, compiled} x {FA3, SDPA} x {torch-XSA, Triton-XSA} x {3-linear, fused-QKV}
#
# Depends on /workspace/work/xsa_triton.py (already there) and FA3 installed.
set -euo pipefail
cd /workspace
echo "=== PHASE 2c ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"

# (a) FA3 module inspection + attention backend bench ----------------------
echo
echo "--- FA3 module inspection ---"
python - <<'PY'
import flash_attn_interface as fa3, inspect, pathlib
print("file:", fa3.__file__)
print("attrs:", [a for a in dir(fa3) if not a.startswith('_')][:25])
# show full source (it's a Python wrapper; only a few KB)
src = pathlib.Path(fa3.__file__).read_text()
print(f"source bytes: {len(src)}")
print("--- source (first 80 lines) ---")
print("\n".join(src.splitlines()[:80]))
PY

echo
echo "--- FA3 vs SDPA attention bench (B=8 T=2048 H=8 Hkv=4 D=64 bf16) ---"
python - <<'PY'
import time, torch, torch.nn.functional as F
from flash_attn_interface import flash_attn_func as fa3

device = torch.device("cuda"); dtype = torch.bfloat16
B, T, H, Hkv, D = 8, 2048, 8, 4, 64
q = torch.randn(B, T, H,   D, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(B, T, Hkv, D, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(B, T, Hkv, D, device=device, dtype=dtype, requires_grad=True)

def fa3_fwd(fwd_only=False):
    q.grad = k.grad = v.grad = None
    out = fa3(q, k, v, causal=True)
    if not fwd_only:
        out.sum().backward()

def sdpa_fwd(fwd_only=False):
    q.grad = k.grad = v.grad = None
    out = F.scaled_dot_product_attention(
        q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
        is_causal=True, enable_gqa=True,
    ).transpose(1,2)
    if not fwd_only:
        out.sum().backward()

def bench(fn, n_warmup=30, n_iter=300):
    for _ in range(n_warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / n_iter

print(f"{'backend':<12} {'fwd us':>10} {'fwd+bwd us':>12}")
print("-"*36)
for lbl, fn in [("FA3", fa3_fwd), ("SDPA", sdpa_fwd)]:
    f = bench(lambda: fn(fwd_only=True))
    t = bench(fn)
    print(f"{lbl:<12} {f:>10.1f} {t:>12.1f}")
PY

# (b) QKV fusion helper + (c) block benchmark ------------------------------
echo
echo "--- Block bench: all variants ---"
python - <<'PY'
import os, sys, time, types, torch, torch.nn.functional as F, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspace/work")
from xsa_triton import xsa_triton as xsa_triton_fn

# Load FA3 baseline module
src = open("/workspace/work/train_gpt_baseline.py").read()
ns = {"__name__": "pg_baseline"}
exec(compile(src, "train_gpt_baseline.py", "exec"), ns)
Block          = ns["Block"]
Rotary         = ns["Rotary"]
CastedLinear   = ns["CastedLinear"]
apply_rotary_emb = ns["apply_rotary_emb"]
fa3_func       = ns["flash_attn_3_func"]  # imported from flash_attn_interface

device = torch.device("cuda"); dtype = torch.bfloat16
torch.manual_seed(0)
B, T, D = 8, 2048, 512
H, KVH = 8, 4

# ----- backend switches ----------------------------------------------------
def sdpa_shim(q, k, v, causal=False):
    gqa = q.size(-2) != k.size(-2)
    return F.scaled_dot_product_attention(
        q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
        is_causal=causal, enable_gqa=gqa,
    ).transpose(1,2)

# ----- fused-QKV forward --------------------------------------------------
def fused_qkv_forward(self, x):
    bsz, seqlen, _ = x.shape
    qkv = self.c_qkv(x)
    q_dim  = self.num_heads    * self.head_dim
    kv_dim = self.num_kv_heads * self.head_dim
    q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
    q = q.reshape(bsz, seqlen, self.num_heads,    self.head_dim)
    k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
    v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))
    cos, sin = self.rotary(seqlen, x.device, q.dtype)
    q = apply_rotary_emb(q, cos, sin, self.rope_dims)
    k = apply_rotary_emb(k, cos, sin, self.rope_dims)
    q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
    y = self._attn_backend(q, k, v, causal=True)
    if self.use_xsa: y = self._xsa_efficient(y, v)
    y = y.reshape(bsz, seqlen, y.size(-2) * y.size(-1))
    return self.proj(y)

def fuse_qkv_into_attn(attn):
    dim    = attn.c_q.weight.size(1)
    q_dim  = attn.num_heads    * attn.head_dim
    kv_dim = attn.num_kv_heads * attn.head_dim
    out_dim = q_dim + 2 * kv_dim
    c_qkv = CastedLinear(dim, out_dim, bias=False)
    c_qkv = c_qkv.to(attn.c_q.weight.device).to(attn.c_q.weight.dtype)
    with torch.no_grad():
        c_qkv.weight.copy_(torch.cat([attn.c_q.weight,
                                       attn.c_k.weight,
                                       attn.c_v.weight], dim=0))
    attn.c_qkv = c_qkv
    del attn.c_q, attn.c_k, attn.c_v
    attn.forward = types.MethodType(fused_qkv_forward, attn)

# ----- build block ---------------------------------------------------------
def build_block(use_triton_xsa, fused_qkv, backend="FA3"):
    blk = Block(
        dim=D, num_heads=H, num_kv_heads=KVH, mlp_mult=4.0,
        rope_base=10000.0, qk_gain_init=5.0, train_seq_len=T,
        layer_idx=7, ln_scale=True,
    ).to(device).to(dtype)
    for p in blk.parameters():
        if p.ndim < 2: p.data = p.data.float()
    blk.parallel = True
    blk.attn.use_xsa = True
    blk.attn.rope_dims = 16
    blk.attn.rotary = Rotary(D // H, base=10000.0, train_seq_len=T, rope_dims=16).to(device)
    # bind chosen attention backend on the module
    if backend == "FA3":
        blk.attn._attn_backend = staticmethod(fa3_func)
    else:
        blk.attn._attn_backend = staticmethod(sdpa_shim)
    # also override the original attn.forward to route through _attn_backend for the
    # 3-linear path (baseline uses flash_attn_3_func directly; we shim it in place)
    if not fused_qkv:
        # patch the existing forward to use _attn_backend instead of flash_attn_3_func
        orig_forward = blk.attn.forward
        # we need a new forward that uses self._attn_backend in place of flash_attn_3_func
        def three_linear_forward(self, x):
            bsz, seqlen, dim_ = x.shape
            q = self.c_q(x).reshape(bsz, seqlen, self.num_heads,    self.head_dim)
            k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            q = apply_rotary_emb(q, cos, sin, self.rope_dims)
            k = apply_rotary_emb(k, cos, sin, self.rope_dims)
            q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
            y = self._attn_backend(q, k, v, causal=True)
            if self.use_xsa: y = self._xsa_efficient(y, v)
            y = y.reshape(bsz, seqlen, dim_); return self.proj(y)
        blk.attn.forward = types.MethodType(three_linear_forward, blk.attn)
    else:
        fuse_qkv_into_attn(blk.attn)
    if use_triton_xsa:
        def _xsa_tr(self, y, v): return xsa_triton_fn(y, v)
        blk.attn._xsa_efficient = types.MethodType(_xsa_tr, blk.attn)
    return blk

# ----- bench wrapper -------------------------------------------------------
def bench_block(callable_, n_warmup=20, n_iter=80):
    x  = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    x0 = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    for _ in range(n_warmup):
        x.grad = None; x0.grad = None
        y = callable_(x, x0); y.sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        x.grad = None; x0.grad = None
        y = callable_(x, x0); y.sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_iter

# ----- grid ----------------------------------------------------------------
grid = []
for backend in ("FA3", "SDPA"):
    for triton_xsa in (False, True):
        for fused in (False, True):
            for compile_ in (False, True):
                grid.append((backend, triton_xsa, fused, compile_))

print(f"shape: B={B} T={T} D={D} H={H} KVH={KVH}  dtype={dtype}")
print(f"{'backend':<5} {'xsa':<8} {'qkv':<9} {'mode':<9} {'ms/iter':>10}")
print("-" * 50)
for backend, triton_xsa, fused, compile_ in grid:
    blk = build_block(use_triton_xsa=triton_xsa, fused_qkv=fused, backend=backend)
    callable_ = torch.compile(blk, dynamic=False, mode="max-autotune-no-cudagraphs") if compile_ else blk
    try:
        ms = bench_block(callable_)
        xsa_lbl = "triton" if triton_xsa else "torch"
        qkv_lbl = "fused" if fused else "3-lin"
        mode_lbl = "compiled" if compile_ else "eager"
        print(f"{backend:<5} {xsa_lbl:<8} {qkv_lbl:<9} {mode_lbl:<9} {ms:>10.3f}")
    except Exception as e:
        print(f"{backend} xsa={triton_xsa} fused={fused} compile={compile_}: FAIL {type(e).__name__}: {str(e)[:80]}")
PY

echo "=== PHASE 2c DONE ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
