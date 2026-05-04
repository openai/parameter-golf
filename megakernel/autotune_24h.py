"""
24-Hour Comprehensive Mega-Kernel AutoSearch
============================================
AI-driven kernel optimization: exhaustively search config space, test
every viable fused-kernel architecture, and produce a ranked map of
what actually works on this GPU.

Saves results to:
  /workspace/megakernel_results/
    results.json         — machine-readable full data
    REPORT.md            — human-readable ranked table
    best_configs.py      — ready-to-paste kernel config
"""
import os, sys, json, time, math, traceback, itertools
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

RESULTS_DIR = Path("/workspace/megakernel_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG = open(RESULTS_DIR / "run.log", "w", buffering=1)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    full = f"[{ts}] {msg}"
    print(full, flush=True)
    LOG.write(full + "\n")

log(f"GPU: {torch.cuda.get_device_name()}")
log(f"Torch: {torch.__version__}  Triton: {triton.__version__}")
log(f"CUDA cap: sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    TMA_AVAIL = True
    log("TMA: AVAILABLE (Hopper HW)")
except ImportError:
    TMA_AVAIL = False
    log("TMA: NOT AVAILABLE")

DEVICE  = "cuda"
DTYPE   = torch.bfloat16
# Competition-realistic token count per GPU (8-GPU run, 589K tokens total)
M_FULL  = 73728
K_DIM   = 512   # d_model
N_MLP   = 1536  # d_mlp (3×)
N_Q     = 512   # d_q
N_K     = 256   # d_kv
N_V     = 256
SCALE   = 1.0 / math.sqrt(3)
EPS     = 1e-6
REPS    = 200   # timing repetitions

ALL_RESULTS = []

# ─── Timing helper ────────────────────────────────────────────────────────────
def bench(fn, warmup=10, reps=REPS):
    for _ in range(warmup):
        try: fn()
        except Exception: return None
    torch.cuda.synchronize()
    try:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000  # ms
    except Exception as e:
        return None

def record(name, config, ms_fused, ms_ref, M, extra=None):
    speedup = ms_ref / ms_fused if (ms_fused and ms_ref) else None
    entry = {
        "name": name,
        "config": config,
        "M": M,
        "ms_fused": ms_fused,
        "ms_ref": ms_ref,
        "speedup": speedup,
        "extra": extra or {},
    }
    ALL_RESULTS.append(entry)
    status = f"{speedup:.3f}x" if speedup else "FAILED"
    log(f"  {name:50s} M={M:6d}  fused={ms_fused:.3f}ms  ref={ms_ref:.3f}ms  speedup={status}")
    save_results()
    return entry

def save_results():
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(ALL_RESULTS, f, indent=2)
    # Also write quick summary
    good = [r for r in ALL_RESULTS if r["speedup"] and r["speedup"] > 1.0]
    good.sort(key=lambda r: -r["speedup"])
    with open(RESULTS_DIR / "REPORT_live.md", "w") as f:
        f.write("# Mega-Kernel Search — Live Results\n\n")
        f.write(f"Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Winning Configs (speedup > 1.0x)\n\n")
        f.write("| Kernel | Config | M | Speedup | ms_fused | ms_ref |\n")
        f.write("|--------|--------|---|---------|----------|--------|\n")
        for r in good[:20]:
            cfg = json.dumps(r["config"])[:60]
            f.write(f"| {r['name']} | {cfg} | {r['M']} | {r['speedup']:.3f}x | {r['ms_fused']:.3f} | {r['ms_ref']:.3f} |\n")
        f.write("\n## All Results\n\n")
        f.write("| Kernel | M | Speedup |\n|--------|---|---------|\n")
        for r in sorted(ALL_RESULTS, key=lambda r: -(r.get("speedup") or 0)):
            sp = f"{r['speedup']:.3f}x" if r.get("speedup") else "FAIL"
            f.write(f"| {r['name']} | {r['M']} | {sp} |\n")


# ─── SECTION 1: Baseline timing at competition scale ──────────────────────────
log("\n" + "="*70)
log("SECTION 1: Baseline Timings (reference for all comparisons)")
log("="*70)

def make_tensors(M, K=K_DIM, N_up=N_MLP, N_q=N_Q, N_k=N_K, N_v=N_V):
    torch.manual_seed(0)
    return {
        "x":    torch.randn(M, K,     dtype=DTYPE, device=DEVICE) * 0.1,
        "up_w": torch.randn(N_up, K,  dtype=DTYPE, device=DEVICE) * 0.02,
        "dn_w": torch.randn(K, N_up,  dtype=DTYPE, device=DEVICE) * 0.02,
        "q_w":  torch.randn(N_q, K,   dtype=DTYPE, device=DEVICE) * 0.02,
        "k_w":  torch.randn(N_k, K,   dtype=DTYPE, device=DEVICE) * 0.02,
        "v_w":  torch.randn(N_v, K,   dtype=DTYPE, device=DEVICE) * 0.02,
    }

for M in [4096, 16384, M_FULL]:
    T = make_tensors(M)
    x, up_w, dn_w, q_w, k_w, v_w = T["x"], T["up_w"], T["dn_w"], T["q_w"], T["k_w"], T["v_w"]

    # Baseline MLP: RMSNorm + LeakyReLU² MLP
    def ref_mlp():
        xn = F.rms_norm(x, (K_DIM,), eps=EPS) * SCALE
        h  = F.leaky_relu(F.linear(xn, up_w), 0.5).square()
        return F.linear(h, dn_w)
    t_ref_mlp = bench(ref_mlp)

    # Baseline QKV: RMSNorm + 3 linears
    def ref_qkv():
        xn = F.rms_norm(x, (K_DIM,), eps=EPS) * SCALE
        return F.linear(xn, q_w), F.linear(xn, k_w), F.linear(xn, v_w)
    t_ref_qkv = bench(ref_qkv)

    log(f"  BASELINE M={M:6d}  mlp={t_ref_mlp:.3f}ms  qkv={t_ref_qkv:.3f}ms")
    record("BASELINE_MLP", {"type": "baseline"}, t_ref_mlp, t_ref_mlp, M)
    record("BASELINE_QKV", {"type": "baseline"}, t_ref_qkv, t_ref_qkv, M)


# ─── SECTION 2: Exhaustive Triton Autotune for RMSNorm + QKV ──────────────────
log("\n" + "="*70)
log("SECTION 2: Triton Autotune — RMSNorm + QKV (ptr-based, all configs)")
log("="*70)

# Build the autotuned QKV kernel — tries all BLOCK/warp/stage combinations
AUTOTUNE_CONFIGS = []
for bm in [32, 64, 128]:
    for bn in [32, 64, 128, 256]:
        for bk in [32, 64, 128]:
            for nw in [2, 4, 8, 16]:
                for ns in [2, 3, 4, 5]:
                    if bm * bk > 16384: continue  # avoid OOM in registers
                    if bn * bk > 32768: continue
                    AUTOTUNE_CONFIGS.append(
                        triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                            num_warps=nw, num_stages=ns
                        )
                    )
log(f"  Total autotune configs: {len(AUTOTUNE_CONFIGS)}")


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def rmsnorm_linear_autotuned(
    x_ptr, w_ptr, out_ptr, inv_rms_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    scale, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    COMPUTE_RMS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc    = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_M,),        dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mk = offs_k < K
        xb = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                     mask=mask_m[:, None] & mk[None, :], other=0.0)
        wb = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                     mask=mask_n[:, None] & mk[None, :], other=0.0)
        acc = tl.dot(xb, tl.trans(wb), acc)
        if COMPUTE_RMS:
            xf = xb.to(tl.float32)
            sum_sq += tl.sum(xf * xf, axis=1)
    if COMPUTE_RMS:
        inv_rms = scale / tl.sqrt(sum_sq / K + eps)
        tl.store(inv_rms_ptr + offs_m, inv_rms.to(tl.float32), mask=mask_m)
    else:
        inv_rms = tl.load(inv_rms_ptr + offs_m, mask=mask_m).to(tl.float32)
    out = (acc * inv_rms[:, None]).to(tl.bfloat16)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             out, mask=mask_m[:, None] & mask_n[None, :])


def autotuned_rmsnorm_linear(x, w, scale, eps, inv_rms_buf=None):
    M, K = x.shape
    N    = w.shape[0]
    out  = torch.empty((M, N), device=x.device, dtype=x.dtype)
    compute_rms = (inv_rms_buf is None)
    if inv_rms_buf is None:
        inv_rms_buf = torch.empty((M,), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    rmsnorm_linear_autotuned[grid](
        x, w, out, inv_rms_buf,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        scale, eps,
        COMPUTE_RMS=compute_rms,
    )
    return out, inv_rms_buf


for M in [4096, 16384, M_FULL]:
    T = make_tensors(M)
    x, up_w, dn_w, q_w, k_w, v_w = T["x"], T["up_w"], T["dn_w"], T["q_w"], T["k_w"], T["v_w"]

    def fused_autotuned_qkv():
        q, inv = autotuned_rmsnorm_linear(x, q_w, SCALE, EPS)
        k, _   = autotuned_rmsnorm_linear(x, k_w, SCALE, EPS, inv)
        v, _   = autotuned_rmsnorm_linear(x, v_w, SCALE, EPS, inv)
        return q, k, v

    def ref_qkv():
        xn = F.rms_norm(x, (K_DIM,), eps=EPS) * SCALE
        return F.linear(xn, q_w), F.linear(xn, k_w), F.linear(xn, v_w)

    t_fused = bench(fused_autotuned_qkv)
    t_ref   = bench(ref_qkv)
    if t_fused:
        # Extract winning config
        best_cfg = rmsnorm_linear_autotuned.best_config
        cfg_str  = str(best_cfg) if best_cfg else "unknown"
        record("K2_QKV_AUTOTUNED", {"best_config": cfg_str, "M": M}, t_fused, t_ref, M)
    else:
        log(f"  K2_QKV_AUTOTUNED M={M}: FAILED")


# ─── SECTION 3: Autotuned RMSNorm + MLP ───────────────────────────────────────
log("\n" + "="*70)
log("SECTION 3: Triton Autotune — RMSNorm + MLP activation (fused 2-op)")
log("="*70)

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def rmsnorm_linear_lrelu2_autotuned(
    x_ptr, w_ptr, out_ptr, aux_ptr, inv_rms_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_om, stride_on,
    scale, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    COMPUTE_RMS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc    = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mk = offs_k < K
        xb = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                     mask=mask_m[:, None] & mk[None, :], other=0.0)
        wb = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                     mask=mask_n[:, None] & mk[None, :], other=0.0)
        acc = tl.dot(xb, tl.trans(wb), acc)
        if COMPUTE_RMS:
            xf = xb.to(tl.float32)
            sum_sq += tl.sum(xf * xf, axis=1)
    if COMPUTE_RMS:
        inv_rms = scale / tl.sqrt(sum_sq / K + eps)
        tl.store(inv_rms_ptr + offs_m, inv_rms.to(tl.float32), mask=mask_m)
    else:
        inv_rms = tl.load(inv_rms_ptr + offs_m, mask=mask_m).to(tl.float32)
    pre = (acc * inv_rms[:, None]).to(tl.bfloat16)
    act = tl.where(pre > 0, pre, 0.5 * pre)
    post = act * act
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             pre, mask=mask_m[:, None] & mask_n[None, :])
    tl.store(aux_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             post, mask=mask_m[:, None] & mask_n[None, :])


def autotuned_rmsnorm_mlp_up(x, w, scale, eps):
    M, K = x.shape; N = w.shape[0]
    pre = torch.empty((M, N), device=x.device, dtype=x.dtype)
    post = torch.empty((M, N), device=x.device, dtype=x.dtype)
    inv_rms_buf = torch.empty((M,), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    rmsnorm_linear_lrelu2_autotuned[grid](
        x, w, pre, post, inv_rms_buf,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        pre.stride(0), pre.stride(1),
        scale, eps,
        COMPUTE_RMS=True,
    )
    return pre, post


for M in [4096, 16384, M_FULL]:
    T = make_tensors(M)
    x, up_w, dn_w = T["x"], T["up_w"], T["dn_w"]

    def fused_autotuned_mlp():
        pre, post = autotuned_rmsnorm_mlp_up(x, up_w, SCALE, EPS)
        return F.linear(post, dn_w)

    def ref_mlp():
        xn = F.rms_norm(x, (K_DIM,), eps=EPS) * SCALE
        h  = F.leaky_relu(F.linear(xn, up_w), 0.5).square()
        return F.linear(h, dn_w)

    t_fused = bench(fused_autotuned_mlp)
    t_ref   = bench(ref_mlp)
    if t_fused:
        best_cfg = rmsnorm_linear_lrelu2_autotuned.best_config
        record("K1_MLP_AUTOTUNED", {"best_config": str(best_cfg)}, t_fused, t_ref, M)
    else:
        log(f"  K1_MLP_AUTOTUNED M={M}: FAILED")


# ─── SECTION 4: TMA-based Kernels (H100 Hopper only) ─────────────────────────
log("\n" + "="*70)
log("SECTION 4: TMA-based Kernels (Hopper sm_90)")
log("="*70)

if TMA_AVAIL:
    # TMA configs — larger blocks enabled by TMA async prefetch
    TMA_AUTOTUNE_CONFIGS = []
    for bm in [64, 128]:
        for bn in [128, 256]:
            for bk in [64]:
                for nw in [4, 8]:
                    for ns in [3, 4, 5]:
                        TMA_AUTOTUNE_CONFIGS.append(
                            triton.Config(
                                {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk},
                                num_warps=nw, num_stages=ns
                            )
                        )

    @triton.autotune(configs=TMA_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def rmsnorm_linear_tma_autotuned(
        a_desc, b_desc, c_desc, inv_rms_ptr,
        M, N, K, scale, eps,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        NUM_SMS: tl.constexpr, COMPUTE_RMS: tl.constexpr,
    ):
        dtype = tl.bfloat16
        start_pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        k_tiles    = tl.cdiv(K, BLOCK_SIZE_K)
        num_tiles  = num_pid_m * num_pid_n
        for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            sum_sq = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
            for ki in range(k_tiles):
                offs_k = ki * BLOCK_SIZE_K
                a = a_desc.load([offs_am, offs_k])
                b = b_desc.load([offs_bn, offs_k])
                accumulator = tl.dot(a, b.T, accumulator)
                if COMPUTE_RMS:
                    af = a.to(tl.float32)
                    sum_sq += tl.sum(af * af, axis=1)
            if COMPUTE_RMS:
                inv_rms = scale / tl.sqrt(sum_sq / K + eps)
                tl.store(inv_rms_ptr + offs_am + tl.arange(0, BLOCK_SIZE_M),
                         inv_rms.to(tl.float32),
                         mask=(offs_am + tl.arange(0, BLOCK_SIZE_M)) < M)
            else:
                inv_rms = tl.load(inv_rms_ptr + offs_am + tl.arange(0, BLOCK_SIZE_M),
                                  mask=(offs_am + tl.arange(0, BLOCK_SIZE_M)) < M).to(tl.float32)
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = (acc0.to(dtype) * inv_rms[:, None]).to(dtype)
            c1 = (acc1.to(dtype) * inv_rms[:, None]).to(dtype)
            c_desc.store([offs_am, pid_n * BLOCK_SIZE_N], c0)
            c_desc.store([offs_am, pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N // 2], c1)

    def tma_rmsnorm_linear(x, w, scale, eps, inv_rms_buf=None):
        M, K = x.shape; N = w.shape[0]
        out  = torch.empty((M, N), device=x.device, dtype=x.dtype)
        compute_rms = (inv_rms_buf is None)
        if inv_rms_buf is None:
            inv_rms_buf = torch.empty((M,), device=x.device, dtype=torch.float32)
        num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        BEST = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}
        a_desc = TensorDescriptor.from_tensor(x, [BEST["BLOCK_SIZE_M"], BEST["BLOCK_SIZE_K"]])
        b_desc = TensorDescriptor.from_tensor(w, [N if N <= 256 else 256, BEST["BLOCK_SIZE_K"]])
        c_desc = TensorDescriptor.from_tensor(out, [BEST["BLOCK_SIZE_M"], BEST["BLOCK_SIZE_N"] // 2])
        grid = lambda _: (min(num_sms,
                              triton.cdiv(M, BEST["BLOCK_SIZE_M"]) *
                              triton.cdiv(N, BEST["BLOCK_SIZE_N"])),)
        rmsnorm_linear_tma_autotuned[grid](
            a_desc, b_desc, c_desc, inv_rms_buf,
            M, N, K, scale, eps,
            BLOCK_SIZE_M=BEST["BLOCK_SIZE_M"], BLOCK_SIZE_N=BEST["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=BEST["BLOCK_SIZE_K"],
            NUM_SMS=num_sms, COMPUTE_RMS=compute_rms,
        )
        return out, inv_rms_buf

    for M in [4096, 16384, M_FULL]:
        T = make_tensors(M)
        x, q_w, k_w, v_w = T["x"], T["q_w"], T["k_w"], T["v_w"]
        def tma_qkv():
            q, inv = tma_rmsnorm_linear(x, q_w, SCALE, EPS)
            k, _   = tma_rmsnorm_linear(x, k_w, SCALE, EPS, inv)
            v, _   = tma_rmsnorm_linear(x, v_w, SCALE, EPS, inv)
            return q, k, v
        def ref_qkv():
            xn = F.rms_norm(x, (K_DIM,), eps=EPS) * SCALE
            return F.linear(xn, q_w), F.linear(xn, k_w), F.linear(xn, v_w)
        try:
            t_fused = bench(tma_qkv)
            t_ref   = bench(ref_qkv)
            record("K2_QKV_TMA", {"type": "TMA"}, t_fused, t_ref, M)
        except Exception as e:
            log(f"  TMA K2 M={M} FAILED: {e}")
else:
    log("  SKIPPED — TMA not available on this GPU")


# ─── SECTION 5: 2-Pass Approach (separate RMS stats + modified GEMM) ──────────
log("\n" + "="*70)
log("SECTION 5: 2-Pass RMSNorm+GEMM — compute stats first, apply in GEMM")
log("="*70)

@triton.autotune(configs=[
    triton.Config({"BLOCK_M": bm, "BLOCK_K": bk}, num_warps=nw)
    for bm in [64, 128, 256]
    for bk in [64, 128, 256]
    for nw in [4, 8, 16]
], key=["M", "K"])
@triton.jit
def compute_inv_rms_kernel(
    x_ptr, inv_rms_ptr, M, K, scale, eps,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        xb = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                     mask=mask_m[:, None] & (offs_k < K)[None, :], other=0.0)
        xf = xb.to(tl.float32)
        sum_sq += tl.sum(xf * xf, axis=1)
    inv_rms = scale / tl.sqrt(sum_sq / K + eps)
    tl.store(inv_rms_ptr + offs_m, inv_rms.to(tl.float32), mask=mask_m)

@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def apply_rms_linear_kernel(
    x_ptr, w_ptr, out_ptr, inv_rms_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M; mask_n = offs_n < N
    # Load inv_rms for this row block once (tiny: BLOCK_M floats)
    inv_rms = tl.load(inv_rms_ptr + offs_m, mask=mask_m, other=1.0).to(tl.bfloat16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mk = offs_k < K
        xb = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                     mask=mask_m[:, None] & mk[None, :], other=0.0)
        xb_scaled = xb * inv_rms[:, None]  # Apply RMSNorm inside GEMM tile
        wb = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                     mask=mask_n[:, None] & mk[None, :], other=0.0)
        acc = tl.dot(xb_scaled, tl.trans(wb), acc)
    out = acc.to(tl.bfloat16)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             out, mask=mask_m[:, None] & mask_n[None, :])


def twopass_rmsnorm_linear(x, w, scale, eps, inv_rms_buf=None):
    M, K = x.shape; N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if inv_rms_buf is None:
        inv_rms_buf = torch.empty((M,), device=x.device, dtype=torch.float32)
        BM, BK = 128, 128
        grid1 = (triton.cdiv(M, BM),)
        compute_inv_rms_kernel[grid1](x, inv_rms_buf, M, K, scale, eps,
                                      BLOCK_M=BM, BLOCK_K=BK)
    grid2 = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    apply_rms_linear_kernel[grid2](
        x, w, out, inv_rms_buf, M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
    )
    return out, inv_rms_buf


for M in [4096, 16384, M_FULL]:
    T = make_tensors(M)
    x, q_w, k_w, v_w, up_w, dn_w = T["x"], T["q_w"], T["k_w"], T["v_w"], T["up_w"], T["dn_w"]

    def twopass_qkv():
        q, inv = twopass_rmsnorm_linear(x, q_w, SCALE, EPS)
        k, _   = twopass_rmsnorm_linear(x, k_w, SCALE, EPS, inv)
        v, _   = twopass_rmsnorm_linear(x, v_w, SCALE, EPS, inv)
        return q, k, v

    def twopass_mlp():
        h, inv = twopass_rmsnorm_linear(x, up_w, SCALE, EPS)
        h = F.leaky_relu(h, 0.5).square()
        return F.linear(h, dn_w)

    def ref_qkv():
        xn = F.rms_norm(x, (K_DIM,), eps=EPS) * SCALE
        return F.linear(xn, q_w), F.linear(xn, k_w), F.linear(xn, v_w)
    def ref_mlp():
        xn = F.rms_norm(x, (K_DIM,), eps=EPS) * SCALE
        return F.linear(F.leaky_relu(F.linear(xn, up_w), 0.5).square(), dn_w)

    try:
        t_qkv = bench(twopass_qkv); t_r_qkv = bench(ref_qkv)
        t_mlp = bench(twopass_mlp); t_r_mlp = bench(ref_mlp)
        record("K2_2PASS_QKV", {"type": "2pass"}, t_qkv, t_r_qkv, M)
        record("K1_2PASS_MLP", {"type": "2pass"}, t_mlp, t_r_mlp, M)
    except Exception as e:
        log(f"  2-pass M={M} FAILED: {e}\n{traceback.format_exc()}")


# ─── SECTION 6: Unified QKV Kernel (single kernel for all 3 projections) ──────
log("\n" + "="*70)
log("SECTION 6: Unified QKV (Q+K+V in one kernel, share K-loop over x)")
log("="*70)

# This kernel reads x ONCE and writes Q, K, V in the same k-loop.
# Total reads: x×1 (75MB) vs x×3 (225MB) in separate approach.
# Downside: larger register pressure, harder to tile N.

@triton.jit
def unified_qkv_kernel(
    x_ptr, q_w_ptr, k_w_ptr, v_w_ptr,
    q_ptr, k_ptr, v_ptr, inv_rms_ptr,
    M, K, N_q, N_k, N_v,
    scale, eps,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_Nq: tl.constexpr, BLOCK_Nk: tl.constexpr, BLOCK_Nv: tl.constexpr,
):
    """
    Each CTA handles BLOCK_M rows of x.
    Inner loops: k-tiles reading x, accumulating into 3 separate (BLOCK_M × BLOCK_N) accumulators.
    This reads x ONCE per row-block but runs 3 GEMMs simultaneously.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)  # col tile (over max(N_q, N_k, N_v))
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Q acc (pid_n indexes into N_q)
    q_offs_n = pid_n * BLOCK_Nq + tl.arange(0, BLOCK_Nq)
    q_mask_n = q_offs_n < N_q
    q_acc = tl.zeros((BLOCK_M, BLOCK_Nq), dtype=tl.float32)

    # K acc
    k_offs_n = pid_n * BLOCK_Nk + tl.arange(0, BLOCK_Nk)
    k_mask_n = k_offs_n < N_k
    k_acc = tl.zeros((BLOCK_M, BLOCK_Nk), dtype=tl.float32)

    # V acc
    v_offs_n = pid_n * BLOCK_Nv + tl.arange(0, BLOCK_Nv)
    v_mask_n = v_offs_n < N_v
    v_acc = tl.zeros((BLOCK_M, BLOCK_Nv), dtype=tl.float32)

    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mk = offs_k < K
        # Load x once
        xb = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                     mask=mask_m[:, None] & mk[None, :], other=0.0)
        # Accumulate sum_sq
        xf = xb.to(tl.float32)
        sum_sq += tl.sum(xf * xf, axis=1)
        # Q GEMM
        qw = tl.load(q_w_ptr + q_offs_n[:, None] * K + offs_k[None, :],
                     mask=q_mask_n[:, None] & mk[None, :], other=0.0)
        q_acc = tl.dot(xb, tl.trans(qw), q_acc)
        # K GEMM
        kw = tl.load(k_w_ptr + k_offs_n[:, None] * K + offs_k[None, :],
                     mask=k_mask_n[:, None] & mk[None, :], other=0.0)
        k_acc = tl.dot(xb, tl.trans(kw), k_acc)
        # V GEMM
        vw = tl.load(v_w_ptr + v_offs_n[:, None] * K + offs_k[None, :],
                     mask=v_mask_n[:, None] & mk[None, :], other=0.0)
        v_acc = tl.dot(xb, tl.trans(vw), v_acc)

    inv_rms = (scale / tl.sqrt(sum_sq / K + eps)).to(tl.bfloat16)
    tl.store(inv_rms_ptr + offs_m, inv_rms.to(tl.float32), mask=mask_m)

    # Write Q, K, V with RMSNorm applied
    q_out = (q_acc * inv_rms[:, None]).to(tl.bfloat16)
    k_out = (k_acc * inv_rms[:, None]).to(tl.bfloat16)
    v_out = (v_acc * inv_rms[:, None]).to(tl.bfloat16)
    tl.store(q_ptr + offs_m[:, None] * N_q + q_offs_n[None, :], q_out,
             mask=mask_m[:, None] & q_mask_n[None, :])
    tl.store(k_ptr + offs_m[:, None] * N_k + k_offs_n[None, :], k_out,
             mask=mask_m[:, None] & k_mask_n[None, :])
    tl.store(v_ptr + offs_m[:, None] * N_v + v_offs_n[None, :], v_out,
             mask=mask_m[:, None] & v_mask_n[None, :])


UNIFIED_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_K": bk,
                   "BLOCK_Nq": bnq, "BLOCK_Nk": bnk, "BLOCK_Nv": bnv},
                  num_warps=nw, num_stages=ns)
    for bm in [32, 64, 128]
    for bk in [32, 64]
    for bnq in [64, 128]
    for bnk in [32, 64]
    for bnv in [32, 64]
    for nw in [4, 8]
    for ns in [2, 3]
    if bm * bk <= 8192  # register limit guard
]

@triton.autotune(configs=UNIFIED_CONFIGS, key=["M", "K", "N_q", "N_k", "N_v"])
@triton.jit
def unified_qkv_autotuned(
    x_ptr, q_w_ptr, k_w_ptr, v_w_ptr,
    q_ptr, k_ptr, v_ptr, inv_rms_ptr,
    M, K, N_q, N_k, N_v, scale, eps,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_Nq: tl.constexpr, BLOCK_Nk: tl.constexpr, BLOCK_Nv: tl.constexpr,
):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); mask_m = offs_m < M
    q_offs_n = pid_n * BLOCK_Nq + tl.arange(0, BLOCK_Nq); qmask = q_offs_n < N_q
    k_offs_n = pid_n * BLOCK_Nk + tl.arange(0, BLOCK_Nk); kmask = k_offs_n < N_k
    v_offs_n = pid_n * BLOCK_Nv + tl.arange(0, BLOCK_Nv); vmask = v_offs_n < N_v
    q_acc = tl.zeros((BLOCK_M, BLOCK_Nq), dtype=tl.float32)
    k_acc = tl.zeros((BLOCK_M, BLOCK_Nk), dtype=tl.float32)
    v_acc = tl.zeros((BLOCK_M, BLOCK_Nv), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K); mk = offs_k < K
        xb = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                     mask=mask_m[:, None] & mk[None, :], other=0.0)
        xf = xb.to(tl.float32); sum_sq += tl.sum(xf * xf, axis=1)
        qw = tl.load(q_w_ptr + q_offs_n[:, None] * K + offs_k[None, :],
                     mask=qmask[:, None] & mk[None, :], other=0.0)
        q_acc = tl.dot(xb, tl.trans(qw), q_acc)
        kw = tl.load(k_w_ptr + k_offs_n[:, None] * K + offs_k[None, :],
                     mask=kmask[:, None] & mk[None, :], other=0.0)
        k_acc = tl.dot(xb, tl.trans(kw), k_acc)
        vw = tl.load(v_w_ptr + v_offs_n[:, None] * K + offs_k[None, :],
                     mask=vmask[:, None] & mk[None, :], other=0.0)
        v_acc = tl.dot(xb, tl.trans(vw), v_acc)
    inv_rms = (scale / tl.sqrt(sum_sq / K + eps)).to(tl.bfloat16)
    tl.store(inv_rms_ptr + offs_m, inv_rms.to(tl.float32), mask=mask_m)
    q_out = (q_acc * inv_rms[:, None]).to(tl.bfloat16)
    k_out = (k_acc * inv_rms[:, None]).to(tl.bfloat16)
    v_out = (v_acc * inv_rms[:, None]).to(tl.bfloat16)
    tl.store(q_ptr + offs_m[:, None] * N_q + q_offs_n[None, :], q_out,
             mask=mask_m[:, None] & qmask[None, :])
    tl.store(k_ptr + offs_m[:, None] * N_k + k_offs_n[None, :], k_out,
             mask=mask_m[:, None] & kmask[None, :])
    tl.store(v_ptr + offs_m[:, None] * N_v + v_offs_n[None, :], v_out,
             mask=mask_m[:, None] & vmask[None, :])


def unified_qkv_fn(x, q_w, k_w, v_w, scale, eps):
    M, K = x.shape
    q = torch.empty((M, N_Q), device=x.device, dtype=x.dtype)
    k = torch.empty((M, N_K), device=x.device, dtype=x.dtype)
    v = torch.empty((M, N_V), device=x.device, dtype=x.dtype)
    inv_rms = torch.empty((M,), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
                         max(triton.cdiv(N_Q, meta["BLOCK_Nq"]),
                             triton.cdiv(N_K, meta["BLOCK_Nk"])))
    unified_qkv_autotuned[grid](
        x, q_w, k_w, v_w, q, k, v, inv_rms,
        M, K, N_Q, N_K, N_V, scale, eps,
    )
    return q, k, v


for M in [4096, 16384, M_FULL]:
    T = make_tensors(M)
    x, q_w, k_w, v_w = T["x"], T["q_w"], T["k_w"], T["v_w"]
    def ref_qkv():
        xn = F.rms_norm(x, (K_DIM,), eps=EPS) * SCALE
        return F.linear(xn, q_w), F.linear(xn, k_w), F.linear(xn, v_w)
    try:
        t_uni = bench(lambda: unified_qkv_fn(x, q_w, k_w, v_w, SCALE, EPS))
        t_ref = bench(ref_qkv)
        record("K3_UNIFIED_QKV", {}, t_uni, t_ref, M)
    except Exception as e:
        log(f"  K3 UNIFIED M={M} FAILED: {e}\n{traceback.format_exc()[:300]}")


# ─── SECTION 7: Extended scaling test ─────────────────────────────────────────
log("\n" + "="*70)
log("SECTION 7: Extended scaling — M from 512 to 131072")
log("="*70)

# For each winning kernel, test across a wide range of M to find where it wins
winning_kernels = [(r["name"], r["config"]) for r in ALL_RESULTS
                   if r.get("speedup") and r["speedup"] >= 1.0 and r["M"] == M_FULL]
winning_kernels = list(dict.fromkeys(n for n, c in winning_kernels))  # deduplicate

if not winning_kernels:
    log("  No winning kernels found yet — testing best candidates anyway")
    winning_kernels = ["K2_QKV_AUTOTUNED", "K2_2PASS_QKV", "K3_UNIFIED_QKV"]

for M in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, M_FULL, 131072]:
    T = make_tensors(M)
    x, q_w, k_w, v_w = T["x"], T["q_w"], T["k_w"], T["v_w"]
    def ref_qkv():
        xn = F.rms_norm(x, (K_DIM,), eps=EPS) * SCALE
        return F.linear(xn, q_w), F.linear(xn, k_w), F.linear(xn, v_w)
    t_ref = bench(ref_qkv)

    try:
        def fn_at():
            q, inv = autotuned_rmsnorm_linear(x, q_w, SCALE, EPS)
            k, _   = autotuned_rmsnorm_linear(x, k_w, SCALE, EPS, inv)
            v, _   = autotuned_rmsnorm_linear(x, v_w, SCALE, EPS, inv)
            return q, k, v
        t = bench(fn_at)
        record("K2_SCALE", {"M": M}, t, t_ref, M)
    except Exception: pass

    try:
        t_2p = bench(lambda: (
            lambda inv=(twopass_rmsnorm_linear(x, q_w, SCALE, EPS))[1]:
            (twopass_rmsnorm_linear(x, q_w, SCALE, EPS, None),
             twopass_rmsnorm_linear(x, k_w, SCALE, EPS, inv),
             twopass_rmsnorm_linear(x, v_w, SCALE, EPS, inv))
        )())
        record("K2_2PASS_SCALE", {"M": M}, t_2p, t_ref, M)
    except Exception: pass


# ─── FINAL REPORT ─────────────────────────────────────────────────────────────
log("\n" + "="*70)
log("FINAL REPORT")
log("="*70)

save_results()

# Ranked by speedup at full M
full_m_results = [r for r in ALL_RESULTS if r["M"] == M_FULL and r.get("speedup")]
full_m_results.sort(key=lambda r: -r["speedup"])

log(f"\nTop 10 kernels at M={M_FULL}:")
for r in full_m_results[:10]:
    log(f"  {r['name']:50s}  {r['speedup']:.3f}x  fused={r['ms_fused']:.3f}ms  ref={r['ms_ref']:.3f}ms")

# Write final markdown report
with open(RESULTS_DIR / "REPORT.md", "w") as f:
    f.write("# Mega-Kernel AutoSearch Results\n\n")
    f.write(f"GPU: {torch.cuda.get_device_name()}\n")
    f.write(f"Torch: {torch.__version__}  Triton: {triton.__version__}\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## Best Kernels at Full Competition Scale (M=73728)\n\n")
    f.write("| Rank | Kernel | Speedup | ms_fused | ms_ref | Notes |\n")
    f.write("|------|--------|---------|----------|--------|-------|\n")
    for i, r in enumerate(full_m_results[:20], 1):
        note = "WINNER" if r["speedup"] > 1.0 else ""
        f.write(f"| {i} | {r['name']} | {r['speedup']:.3f}x | {r['ms_fused']:.3f} | {r['ms_ref']:.3f} | {note} |\n")
    f.write("\n## Scaling Analysis\n\n")
    scale_results = [r for r in ALL_RESULTS if "SCALE" in r["name"]]
    f.write("| M | Kernel | Speedup |\n|---|--------|---------|\n")
    for r in sorted(scale_results, key=lambda r: r["M"]):
        sp = f"{r['speedup']:.3f}x" if r.get("speedup") else "FAIL"
        f.write(f"| {r['M']} | {r['name']} | {sp} |\n")
    f.write("\n## Recommendation\n\n")
    winners = [r for r in full_m_results if r.get("speedup") and r["speedup"] > 1.0]
    if winners:
        best = winners[0]
        f.write(f"**USE {best['name']}** — {best['speedup']:.3f}x speedup at M={M_FULL}\n\n")
        f.write(f"Config: `{json.dumps(best['config'])}`\n")
    else:
        f.write("No kernel beats baseline at full scale. **Recommendation: Keep unfused path.**\n")
        f.write("Possible causes: memory-bound ops already near peak, WGMMA disrupted by extra sum_sq work.\n")

log("\nDone. Results written to /workspace/megakernel_results/")
log(f"  results.json  — {len(ALL_RESULTS)} records")
log(f"  REPORT.md     — ranked table")
LOG.close()
