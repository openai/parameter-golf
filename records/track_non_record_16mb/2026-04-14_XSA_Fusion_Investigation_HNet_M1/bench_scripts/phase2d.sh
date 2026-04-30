#!/usr/bin/env bash
# Phase 2d:
#   - Fix FA3 staticmethod bug (bind backend via closure, not staticmethod).
#   - Verify QKV-fusion numerical equivalence.
#   - Robust benchmark: compile ALL variants first, warm each, then measure
#     each TWICE in a fixed order to detect drift. Focus on compiled numbers.
set -euo pipefail
cd /workspace
echo "=== PHASE 2d ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"

python - <<'PY'
import os, sys, time, types, warnings, torch, torch.nn.functional as F
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspace/work")
from xsa_triton import xsa_triton as xsa_triton_fn

# Load FA3 baseline module
src = open("/workspace/work/train_gpt_baseline.py").read()
ns = {"__name__": "pg_baseline"}
exec(compile(src, "train_gpt_baseline.py", "exec"), ns)
Block            = ns["Block"]
Rotary           = ns["Rotary"]
CastedLinear     = ns["CastedLinear"]
apply_rotary_emb = ns["apply_rotary_emb"]
fa3_func         = ns["flash_attn_3_func"]

device = torch.device("cuda"); dtype = torch.bfloat16
torch.manual_seed(0)

B, T, D = 8, 2048, 512
H, KVH = 8, 4

# --------------------------------------------------------------------------
# Backend functions (plain callables, no staticmethod)
# --------------------------------------------------------------------------
def sdpa_backend(q, k, v, causal=False):
    gqa = q.size(-2) != k.size(-2)
    return F.scaled_dot_product_attention(
        q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
        is_causal=causal, enable_gqa=gqa,
    ).transpose(1,2)

def fa3_backend(q, k, v, causal=False):
    return fa3_func(q, k, v, causal=causal)

# --------------------------------------------------------------------------
# Forwards: closures capture backend & xsa functions (avoids staticmethod issue)
# --------------------------------------------------------------------------
def make_three_linear_forward(attn_backend, use_triton_xsa):
    def forward(self, x):
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
        y = attn_backend(q, k, v, causal=True)
        if self.use_xsa:
            y = xsa_triton_fn(y, v) if use_triton_xsa else self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim_); return self.proj(y)
    return forward

def make_fused_qkv_forward(attn_backend, use_triton_xsa):
    def forward(self, x):
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
        y = attn_backend(q, k, v, causal=True)
        if self.use_xsa:
            y = xsa_triton_fn(y, v) if use_triton_xsa else self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, y.size(-2) * y.size(-1))
        return self.proj(y)
    return forward

def fuse_qkv_weights(attn):
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

# --------------------------------------------------------------------------
# Numerical equivalence check: fused-QKV vs 3-linear with identical weights
# --------------------------------------------------------------------------
print("--- QKV fusion numerical check ---")
torch.manual_seed(0)
blk3 = Block(dim=D, num_heads=H, num_kv_heads=KVH, mlp_mult=4.0,
             rope_base=10000.0, qk_gain_init=5.0, train_seq_len=T,
             layer_idx=7, ln_scale=True).to(device).to(dtype)
blk3.parallel = True; blk3.attn.use_xsa = True
blk3.attn.rope_dims = 16
blk3.attn.rotary = Rotary(D // H, base=10000.0, train_seq_len=T, rope_dims=16).to(device)
blk3.attn.forward = types.MethodType(
    make_three_linear_forward(sdpa_backend, use_triton_xsa=False), blk3.attn)

# Mirror blk3 into a fused-QKV copy
import copy as _copy
blkF = _copy.deepcopy(blk3)
fuse_qkv_weights(blkF.attn)
blkF.attn.forward = types.MethodType(
    make_fused_qkv_forward(sdpa_backend, use_triton_xsa=False), blkF.attn)

x  = torch.randn(2, 128, D, device=device, dtype=dtype)
x0 = torch.randn(2, 128, D, device=device, dtype=dtype)
with torch.no_grad():
    y3 = blk3(x, x0)
    yF = blkF(x, x0)
fwd_err = (y3 - yF).abs().max().item()
print(f"  fwd max err (3-lin vs fused): {fwd_err:.3e}  (bf16 tol ~1e-2)")

# --------------------------------------------------------------------------
# Build + compile all variants, then bench each twice
# --------------------------------------------------------------------------
def build_variant(backend_name, use_triton_xsa, fused_qkv):
    backend = fa3_backend if backend_name == "FA3" else sdpa_backend
    torch.manual_seed(0)
    blk = Block(dim=D, num_heads=H, num_kv_heads=KVH, mlp_mult=4.0,
                rope_base=10000.0, qk_gain_init=5.0, train_seq_len=T,
                layer_idx=7, ln_scale=True).to(device).to(dtype)
    for p in blk.parameters():
        if p.ndim < 2: p.data = p.data.float()
    blk.parallel = True; blk.attn.use_xsa = True
    blk.attn.rope_dims = 16
    blk.attn.rotary = Rotary(D // H, base=10000.0, train_seq_len=T, rope_dims=16).to(device)
    if fused_qkv:
        fuse_qkv_weights(blk.attn)
        blk.attn.forward = types.MethodType(
            make_fused_qkv_forward(backend, use_triton_xsa), blk.attn)
    else:
        blk.attn.forward = types.MethodType(
            make_three_linear_forward(backend, use_triton_xsa), blk.attn)
    return blk

variants = []
for backend in ("SDPA", "FA3"):
    for use_triton_xsa in (False, True):
        for fused in (False, True):
            variants.append((backend, use_triton_xsa, fused))

# Build + compile + warm up each variant
print("\n--- Compiling + warming up all variants (this takes a few min) ---")
x  = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
x0 = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
compiled_blocks = {}
for i, (backend, triton_xsa, fused) in enumerate(variants):
    tag = f"{backend}/{'triton' if triton_xsa else 'torch '}XSA/{'fused' if fused else '3-lin'}"
    t0 = time.perf_counter()
    blk = build_variant(backend, triton_xsa, fused)
    cfn = torch.compile(blk, dynamic=False, mode="max-autotune-no-cudagraphs")
    try:
        for _ in range(25):
            x.grad = None; x0.grad = None
            y = cfn(x, x0); y.sum().backward()
        torch.cuda.synchronize()
        compiled_blocks[(backend, triton_xsa, fused)] = cfn
        print(f"  [{i+1}/{len(variants)}] {tag}  warmup OK in {time.perf_counter()-t0:.1f}s")
    except Exception as e:
        print(f"  [{i+1}/{len(variants)}] {tag}  FAIL: {type(e).__name__}: {str(e)[:100]}")

# --------------------------------------------------------------------------
# Bench each variant TWICE, report both to see drift
# --------------------------------------------------------------------------
def bench_block(callable_, n_iter=150):
    # no warmup here, we already warmed up
    x.grad = None; x0.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        x.grad = None; x0.grad = None
        y = callable_(x, x0); y.sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_iter

print("\n--- Compiled block bench (two measurements; ms/iter) ---")
print(f"shape: B={B} T={T} D={D} H={H} KVH={KVH}")
print(f"{'variant':<35} {'run1':>9} {'run2':>9} {'drift%':>8}")
print("-" * 63)
# First pass
measurements = {}
for (backend, triton_xsa, fused), cfn in compiled_blocks.items():
    ms1 = bench_block(cfn)
    measurements[(backend, triton_xsa, fused)] = [ms1]
# Second pass (different iteration order — reversed — to detect thermal/cache drift)
for (backend, triton_xsa, fused), cfn in reversed(list(compiled_blocks.items())):
    ms2 = bench_block(cfn)
    measurements[(backend, triton_xsa, fused)].append(ms2)

# Print in logical order
best = (float("inf"), None)
baseline = None
for backend in ("SDPA", "FA3"):
    for triton_xsa in (False, True):
        for fused in (False, True):
            key = (backend, triton_xsa, fused)
            if key not in measurements: continue
            xsa_lbl = "triton-XSA" if triton_xsa else "torch-XSA "
            qkv_lbl = "fused-QKV" if fused else "3-lin-QKV"
            tag = f"{backend:<4} {xsa_lbl} {qkv_lbl}"
            ms1, ms2 = measurements[key]
            drift = abs(ms1 - ms2) / min(ms1, ms2) * 100
            print(f"{tag:<35} {ms1:>9.3f} {ms2:>9.3f} {drift:>7.1f}%")
            use_ms = min(ms1, ms2)
            if backend == "SDPA" and not triton_xsa and not fused:
                baseline = use_ms
            if use_ms < best[0]:
                best = (use_ms, key)

print("\n--- Summary ---")
if baseline is not None and best[1] is not None:
    best_ms, best_key = best
    print(f"SDPA baseline (torch-XSA, 3-lin-QKV):  {baseline:.3f} ms/iter")
    print(f"Best variant:  {best_key}:  {best_ms:.3f} ms/iter")
    print(f"Speedup:  {(baseline/best_ms - 1)*100:.1f}%")

print("=== PHASE 2d DONE ===")
import datetime; print(datetime.datetime.utcnow().isoformat() + "Z")
PY
