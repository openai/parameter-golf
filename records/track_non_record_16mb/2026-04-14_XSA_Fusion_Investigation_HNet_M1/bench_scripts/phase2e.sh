#!/usr/bin/env bash
# Phase 2e: robust block microbench.
#   - torch._dynamo.reset() + empty_cache() between variants (fixes FA3 recompile-fallback bug).
#   - Thermal prime (GPU hot) before each measurement.
#   - CUDA events for per-iter timing; 300 samples; report p10/p50/p90.
set -euo pipefail
cd /workspace
echo "=== PHASE 2e ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"

python - <<'PY'
import os, sys, time, types, warnings, torch, torch.nn.functional as F, torch._dynamo
warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspace/work")
from xsa_triton import xsa_triton as xsa_triton_fn

src = open("/workspace/work/train_gpt_baseline.py").read()
ns = {"__name__": "pg_baseline"}
exec(compile(src, "train_gpt_baseline.py", "exec"), ns)
Block = ns["Block"]; Rotary = ns["Rotary"]; CastedLinear = ns["CastedLinear"]
apply_rotary_emb = ns["apply_rotary_emb"]; fa3_func = ns["flash_attn_3_func"]

device = torch.device("cuda"); dtype = torch.bfloat16
B, T, D = 8, 2048, 512; H, KVH = 8, 4

def sdpa_backend(q, k, v, causal=False):
    return F.scaled_dot_product_attention(
        q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
        is_causal=causal, enable_gqa=(q.size(-2)!=k.size(-2)),
    ).transpose(1,2)

def fa3_backend(q, k, v, causal=False): return fa3_func(q, k, v, causal=causal)

def make_three_linear_forward(attn_backend, use_triton_xsa):
    def forward(self, x):
        bsz, seqlen, dim_ = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads,    self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),)); k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims); k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = attn_backend(q, k, v, causal=True)
        if self.use_xsa: y = xsa_triton_fn(y, v) if use_triton_xsa else self._xsa_efficient(y, v)
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
        q = F.rms_norm(q, (q.size(-1),)); k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims); k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = attn_backend(q, k, v, causal=True)
        if self.use_xsa: y = xsa_triton_fn(y, v) if use_triton_xsa else self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, y.size(-2) * y.size(-1)); return self.proj(y)
    return forward

def fuse_qkv_weights(attn):
    dim = attn.c_q.weight.size(1)
    q_dim = attn.num_heads * attn.head_dim; kv_dim = attn.num_kv_heads * attn.head_dim
    c_qkv = CastedLinear(dim, q_dim + 2*kv_dim, bias=False)
    c_qkv = c_qkv.to(attn.c_q.weight.device).to(attn.c_q.weight.dtype)
    with torch.no_grad():
        c_qkv.weight.copy_(torch.cat([attn.c_q.weight, attn.c_k.weight, attn.c_v.weight], dim=0))
    attn.c_qkv = c_qkv
    del attn.c_q, attn.c_k, attn.c_v

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
        blk.attn.forward = types.MethodType(make_fused_qkv_forward(backend, use_triton_xsa), blk.attn)
    else:
        blk.attn.forward = types.MethodType(make_three_linear_forward(backend, use_triton_xsa), blk.attn)
    return blk

# --- thermal primer: long GEMM loop to get GPU to turbo ---
def thermal_prime(seconds=3.0):
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        c = a @ b
    torch.cuda.synchronize()

# --- per-variant measurement ---
def measure_variant(cfn, n_warmup=30, n_samples=300):
    x  = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    x0 = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    for _ in range(n_warmup):
        x.grad = None; x0.grad = None
        y = cfn(x, x0); y.sum().backward()
    torch.cuda.synchronize()

    thermal_prime(2.0)

    samples_ms = []
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_samples)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(n_samples)]
    for i in range(n_samples):
        x.grad = None; x0.grad = None
        starts[i].record()
        y = cfn(x, x0); y.sum().backward()
        ends[i].record()
    torch.cuda.synchronize()
    samples_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    samples_ms.sort()
    return samples_ms

variants = [
    ("SDPA", False, False),  # baseline
    ("SDPA", False, True),   # +fused-QKV
    ("SDPA", True,  False),  # +triton-XSA
    ("SDPA", True,  True),   # +both
    ("FA3",  False, False),
    ("FA3",  False, True),
    ("FA3",  True,  False),
    ("FA3",  True,  True),
]

print("--- Bench (300 samples per variant, CUDA events, dynamo reset between) ---")
print(f"shape: B={B} T={T} D={D} H={H} KVH={KVH}")
print(f"{'variant':<32} {'p10':>7} {'p50':>7} {'p90':>7} {'min':>7}")
print("-" * 64)

results = {}
for backend, triton_xsa, fused in variants:
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    blk = build_variant(backend, triton_xsa, fused)
    cfn = torch.compile(blk, dynamic=False, mode="max-autotune-no-cudagraphs")
    try:
        s = measure_variant(cfn)
        n = len(s)
        p10 = s[int(n*0.10)]; p50 = s[int(n*0.50)]; p90 = s[int(n*0.90)]; smin = s[0]
        xsa_lbl = "triton-XSA" if triton_xsa else "torch-XSA "
        qkv_lbl = "fused-QKV" if fused else "3-lin-QKV"
        label = f"{backend:<4} {xsa_lbl} {qkv_lbl}"
        print(f"{label:<32} {p10:>7.3f} {p50:>7.3f} {p90:>7.3f} {smin:>7.3f}")
        results[(backend, triton_xsa, fused)] = (p10, p50, p90, smin)
    except Exception as e:
        print(f"FAIL {backend} xsa={triton_xsa} fused={fused}: {type(e).__name__}: {str(e)[:80]}")

print("\n--- Summary (using p50) ---")
base_key = ("SDPA", False, False)
base_p50 = results.get(base_key, (None, None, None, None))[1]
if base_p50:
    print(f"Baseline (SDPA torch-XSA 3-lin-QKV) p50: {base_p50:.3f} ms")
    print(f"\n{'variant':<32} {'Δ p50':>8} {'speedup':>8}")
    for k, (p10, p50, p90, smin) in sorted(results.items(), key=lambda kv: kv[1][1]):
        xsa_lbl = "triton-XSA" if k[1] else "torch-XSA "
        qkv_lbl = "fused-QKV" if k[2] else "3-lin-QKV"
        tag = f"{k[0]:<4} {xsa_lbl} {qkv_lbl}"
        delta = p50 - base_p50
        spd = (base_p50 / p50 - 1) * 100
        print(f"{tag:<32} {delta:>+7.3f} {spd:>+7.1f}%")
print("\n=== PHASE 2e DONE ===")
PY
date -u +"%Y-%m-%dT%H:%M:%SZ"
