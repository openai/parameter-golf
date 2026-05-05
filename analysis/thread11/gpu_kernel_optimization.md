# Thread 11: H100 GPU Kernel Optimization Deep-Dive
## CUDA/Compute Analysis for Parameter Golf

**Date**: 2026-03-18  
**Hardware**: NVIDIA H100 80GB (PCIe/SXM), CUDA 13.0, PyTorch 2.10  
**Model**: 9-layer transformer, dim=512, vocab=1024, seq_len=1024, 17M params  
**Baseline**: 43ms/step on 8×H100, 13,780 steps in 10 min → val_bpb=1.2244

---

## 1. H100 ROOFLINE ANALYSIS: MEMORY-BOUND AT dim=512

### H100 SXM Peak Specs
| Metric | Value |
|--------|-------|
| BF16 Tensor Core | 1,979 TFLOPS |
| FP8 (e4m3) Tensor Core | 3,958 TFLOPS |
| HBM3 Bandwidth | 3,350 GB/s |
| BF16 Roofline Crossover | 590.7 FLOP/byte |
| FP8 Roofline Crossover | 1,181 FLOP/byte |
| L2 Cache | 50 MB |
| SM Count | 132 |

### Arithmetic Intensity Per Operation

For a GEMM Y = X @ W^T with X[M, K], W[N, K]:
- FLOPs = 2·M·K·N
- Bytes_read = (M·K + K·N) · bytes_per_elem
- Bytes_write = M·N · bytes_per_elem  
- OI = FLOPs / Total_bytes

With 8 GPUs, per-GPU batch = 524,288 / 8 = 65,536 tokens → M = 65,536.

| Layer Operation | M | K | N | FLOPs (G) | Bytes BF16 (MB) | OI (BF16) | Bound |
|----------------|---|---|---|-----------|-----------------|-----------|-------|
| Q proj (512→512) | 65536 | 512 | 512 | 34.4 | 134.7 | **255** | MEMORY |
| K proj (512→256) | 65536 | 512 | 256 | 17.2 | 100.9 | **170** | MEMORY |
| V proj (512→256) | 65536 | 512 | 256 | 17.2 | 100.9 | **170** | MEMORY |
| Out proj (512→512) | 65536 | 512 | 512 | 34.4 | 134.7 | **255** | MEMORY |
| MLP fc (512→1024) | 65536 | 512 | 1024 | 68.7 | 202.4 | **340** | MEMORY |
| MLP proj (1024→512) | 65536 | 1024 | 512 | 68.7 | 202.4 | **340** | MEMORY |
| LM head (512→1024) | 65536 | 512 | 1024 | 68.7 | 202.4 | **340** | MEMORY |

**CRITICAL FINDING: Every single matmul is MEMORY-BOUND at dim=512.**  
All OIs (170-340) are below the BF16 crossover (590.7). We're utilizing at most
340/590.7 = **57.5%** of peak compute.

### Why dim=512 is the Problem

The OI for large-M GEMM simplifies to:
```
OI ≈ 2·K·N / (K + N) / bytes_per_element
```

For BF16 (2 bytes/elem):
- K=N=512: OI = 2·512·512 / (512+512) / 2 = 256
- K=N=768: OI = 2·768·768 / (768+768) / 2 = 384
- K=N=1024: OI = 2·1024·1024 / (1024+1024) / 2 = 512
- K=N=2048: OI = 2·2048·2048 / (2048+2048) / 2 = 1024 ← COMPUTE-BOUND!

**At dim=512, you'd need 3.4x wider model (dim=1740) to become compute-bound in BF16.**

### Model FLOPs Utilization (MFU)

Per-GPU forward pass FLOPs (9 layers):
- Attention linears per layer: 34.4 + 17.2 + 17.2 + 34.4 = 103.2 GFLOPs
- FlashAttention per layer: ~68.7 GFLOPs (2·B·T²·D·H estimated)
- MLP per layer: 68.7 + 68.7 = 137.4 GFLOPs
- Per layer total: ~309 GFLOPs
- 9 layers + LM head: 2.85 TFLOPs

Forward + backward + optimizer ≈ 3.5× forward = **10.0 TFLOPs/step**

At 1,979 TFLOPS peak, pure compute time = 10.0 / 1979 = **5.1 ms**

Actual step time = 43.5 ms → **MFU = 11.6%**

**Where does the other 88.4% go?**
| Component | Est. Time (ms) | % of Step |
|-----------|----------------|-----------|
| Memory-bound matmul inefficiency | 12-15 | 28-34% |
| Kernel launch overhead | 8-12 | 18-28% |
| Elementwise ops (norms, scales, RoPE) | 4-6 | 9-14% |
| DDP gradient all-reduce | 2-3 | 5-7% |
| Muon optimizer (NS iterations + allreduce) | 4-6 | 9-14% |
| CUDA sync, scheduling, misc | 3-5 | 7-11% |

---

## 2. FP8 MATMUL: DOES IT HELP?

### The Theory

FP8 on H100 provides:
- 2× TFLOPS (3,958 vs 1,979)
- 1 byte/elem vs 2 bytes/elem → halved memory traffic

But since we're **memory-bound**, the 2× compute doesn't matter — we're not hitting
the compute ceiling. However, FP8 DOES reduce memory traffic:

| Operation | BF16 Bytes (MB) | FP8 Reads + BF16 Writes (MB) | Speedup |
|-----------|-----------------|------------------------------|---------|
| Q proj | 134.7 | 100.9 (inputs FP8) + 67.1 (output BF16) = 101.2 | 1.33× |
| MLP fc | 202.4 | 134.7 + 134.2 = 168.2* | 1.20× |
| MLP proj | 202.4 | 201.4 + 67.1 = 168.2* | 1.20× |

*Note: output is BF16 so writes don't shrink. Only reads benefit from FP8 quantization.

**Net FP8 speedup for matmuls: ~1.2-1.3× (not 2×!)** because output writes are still BF16.

### modded-nanogpt FP8 Implementation

modded-nanogpt uses FP8 ONLY for the lm_head (the final embedding→logit projection), using
`torch._scaled_mm` with static scaling factors:

```python
# Static scales (no dynamic calibration)
x_s = torch.tensor(100/448, dtype=torch.float32, device=device)    # activation scale
w_s = torch.tensor(1.6/448, dtype=torch.float32, device=device)    # weight scale
grad_s = torch.tensor(grad_scale * 0.75/448, dtype=torch.float32, device=device)  # grad scale

# Forward: float8_e4m3fn (higher precision for forward)
x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
w_f8 = w_f8.T.contiguous().T  # column-major layout for scaled_mm
out = torch._scaled_mm(x_f8, w_f8, out_dtype=torch.bfloat16,
                       scale_a=x_s, scale_b=w_s, use_fast_accum=True)

# Backward: float8_e5m2 (wider range for gradients)
grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
```

### Applicability to Parameter Golf

**With vocab=1024 and dim=512, the lm_head matmul is [65536, 512] × [512, 1024].**
This is a 68.7 GFLOP operation. At 43ms/step total, it represents ~2% of step time.

**Verdict: FP8 for lm_head saves < 0.5ms/step. NOT WORTH THE IMPLEMENTATION COMPLEXITY.**

For ALL matmuls (attention + MLP), FP8 would save ~3-5ms from reduced memory traffic.
But this requires:
- Custom autograd functions with `torch._scaled_mm`
- Static or dynamic scaling calibration
- Careful gradient handling with `float8_e5m2`
- May break `torch.compile(fullgraph=True)` if using custom ops

**Recommendation: FP8 for all linears = 3-5ms savings (~8-12% speedup). Worth it ONLY if
we can maintain `fullgraph=True` compatibility. Test with `torchao.float8` which provides
compile-friendly FP8 wrappers.**

```python
# Best approach: torchao.float8 (compile-friendly)
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
config = Float8LinearConfig(
    cast_config_input=CastConfig(scaling_type=ScalingType.DYNAMIC),
    cast_config_weight=CastConfig(scaling_type=ScalingType.DYNAMIC),
)
convert_to_float8_training(base_model, config=config)
# Then torch.compile as normal
```

**Expected speedup: 8-12% (3-5ms) with torchao. Higher risk if breaking compile.**

---

## 3. TORCH.COMPILE MODES

### Current Configuration
```python
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
```
No `mode` specified = default mode. This does:
- Trace the forward graph
- Fuse element-wise operations
- Generate Triton/CUDA kernels via Inductor
- No CUDA graph capture
- No exhaustive kernel autotuning

### Mode Comparison

| Mode | Compile Time | Runtime | CUDA Graphs | Autotuning |
|------|-------------|---------|-------------|------------|
| `default` | ~60s | Baseline | No | Basic |
| `reduce-overhead` | ~90s | **-10-15%** | **Yes** | Basic |
| `max-autotune` | ~3-5 min | -5-10% | No | **Exhaustive** |
| `max-autotune-no-cudagraphs` | ~3-5 min | -5-10% | No | Exhaustive |

### Why `reduce-overhead` is the Best Choice

For our model (dim=512, memory-bound), **kernel launch overhead is a major bottleneck**.
Each forward pass launches hundreds of CUDA kernels:
- 9 layers × (4 matmuls + 2 norms + rotary + scaling + attention + MLP activation) = ~100 kernels
- Plus backward pass (~200 kernels)
- Each kernel launch costs 5-20μs on host side

With ~300 kernels per step, launch overhead = 300 × 10μs = **3ms per step**.
CUDA graphs capture the entire sequence and replay it in one launch = **~0.01ms**.

**`reduce-overhead` wraps the compiled graph in CUDA graphs automatically.**

### BUT: CUDA Graphs + DDP Conflict

CUDA graphs cannot capture NCCL collectives (all-reduce). The DDP wrapper adds
an all-reduce hook after the backward pass. Options:

1. **Use `reduce-overhead` WITHOUT DDP**: Only works for single GPU.
2. **Segment the graph**: Forward in CUDA graph, then eager backward+allreduce.
   This is what `reduce-overhead` attempts automatically.
3. **Use FSDP2**: Better CUDA graph support than DDP.

**In practice with DDP**: `reduce-overhead` can still capture the forward pass
as a CUDA graph, saving ~50% of the launch overhead.

### Recommended Configuration
```python
# Best for our setup: reduce-overhead with DDP-aware segmentation
compiled_model = torch.compile(
    base_model,
    dynamic=False,
    fullgraph=True,
    mode='reduce-overhead',
)

# Also set Inductor configs for better fusion:
import torch._inductor.config as inductor_config
inductor_config.coordinate_descent_tuning = True
inductor_config.triton.unique_kernel_names = True
inductor_config.fx_graph_cache = True  # Cache compiled graphs
```

**Expected speedup: 5-8% with DDP (forward CUDA graph + better fusion).**
**10-15% without DDP (full CUDA graph capture).**

### compiled_autograd (Compile the Backward Too)

By default, torch.compile ONLY compiles the forward pass. The backward pass
uses eager PyTorch autograd. Enabling compiled_autograd extends compilation:

```python
import torch._dynamo
torch._dynamo.config.compiled_autograd = True
```

This gives the backward pass the same kernel fusion benefits. Since backward
is ~2× the forward compute, this could save an additional 5-10%.

**Risk**: Experimental feature. May conflict with DDP hooks. Test carefully.

---

## 4. FLASH ATTENTION BACKEND ANALYSIS

### Current Configuration
```python
enable_cudnn_sdp(False)     # cuDNN attention disabled
enable_flash_sdp(True)      # FlashAttention-2 enabled
enable_mem_efficient_sdp(False)
enable_math_sdp(False)
```

### Backend Comparison on H100

| Backend | Speed (relative) | H100 Hopper Features | GQA Support | Notes |
|---------|-----------------|---------------------|-------------|-------|
| FlashAttention-2 (flash_sdp) | 1.0× | No Hopper-specific | Yes | Current |
| cuDNN SDP (cudnn_sdp) | 0.8-1.3× | **Yes** (wgMMA, TMA) | Yes (PyTorch 2.5+) | Highly config-dependent |
| FlashAttention-3 (external) | **1.5-2.0×** | **Yes** (async matmul+softmax) | Yes | Requires separate install |
| Math SDP | 0.3× | No | Yes | Reference, never use |

### cuDNN SDP on H100: The Key Details

PyTorch's cuDNN SDP backend (cudnn_sdp) uses NVIDIA's cuDNN library which has
H100-specific kernels using:
- **wgMMA** (warp group matrix-multiply-accumulate) — H100's most efficient matmul API
- **TMA** (Tensor Memory Accelerator) — hardware-assisted async memory loads
- **GQA native support** since cuDNN 9.0

**However**, cuDNN SDP has strict requirements:
- head_dim must be ≤ 256 and divisible by 8 → **Our head_dim=64 is fine ✓**
- QKV must be in specific memory layout
- May not work with custom q_gain scaling before attention

For our model (head_dim=64, seq_len=1024, GQA 8:4):

```python
# Change to try cuDNN SDP:
enable_cudnn_sdp(True)    # Enable cuDNN
enable_flash_sdp(True)    # Keep flash as fallback
enable_mem_efficient_sdp(False)
enable_math_sdp(False)
```

PyTorch will automatically select the fastest backend that supports the configuration.
With both enabled, it benchmarks and chooses the winner.

**Expected speedup from cuDNN SDP: 0-15%** (highly variable, must benchmark).

### FlashAttention-3: The Real Prize

FlashAttention-3 (FA3) is what modded-nanogpt uses for its record runs. It provides:
- **Asynchronous matmul + softmax pipelining** using H100's async Tensor Cores
- **Intra-warpgroup overlapping** — compute softmax of tile N while computing matmul of tile N+1
- **FP8 attention** option for even more throughput

FA3 requires:
- Separate installation (`pip install flash-attn` or building from source)
- H100/H200 (Hopper SM90 only)
- Direct API call (not through PyTorch SDPA):

```python
from flash_attn import flash_attn_func
# or FA3 specifically:
from flash_attn_interface import flash_attn_func as flash_attn_3

y = flash_attn_func(q, k, v, causal=True, softmax_scale=scale)
```

**BUT**: Using FA3 directly breaks `torch.compile(fullgraph=True)` unless
registered as a custom op. modded-nanogpt solves this with:
```python
flash_attn_interface = get_kernel('varunneal/flash-attention-3').flash_attn_interface
```

**Expected speedup from FA3: 15-30% on attention** → ~5-10% on total step time.
**Risk: HIGH** — requires separate build, may break compile, adds dependency.

### Recommendation

1. **Immediate**: Enable cuDNN SDP alongside flash_sdp (let PyTorch pick winner). Zero risk.
2. **If we need more speed**: Install FA3 and register as custom op.

```python
# Step 1: Easy, zero-risk
enable_cudnn_sdp(True)   # Changed from False to True
enable_flash_sdp(True)
enable_mem_efficient_sdp(False)
enable_math_sdp(False)
```

---

## 5. OPERATOR FUSION OPPORTUNITIES

### What torch.compile Already Fuses

torch.compile with Inductor automatically fuses:
- Element-wise chains: RMSNorm → scale → reshape (fused into one kernel)
- Softcap: logits / cap → tanh → * cap (fused into one kernel)
- Activation: relu → square (fused)
- Cast operations: weight.to(bf16) fused with subsequent matmul

**What it CANNOT fuse:**
- Cross-entropy reduction (writes intermediate logits, then reduces)
- Matmul + element-wise (matmul uses tensor cores, element-wise uses CUDA cores)
- RMSNorm + subsequent matmul (different compute patterns)

### Fused Softcap + Cross-Entropy

The current code:
```python
logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
return F.cross_entropy(logits.float(), targets, reduction="mean")
```

This requires:
1. Materializing full logits tensor: [65536, 1024] × 4 bytes (float32) = **256 MB**
2. Then computing log_softmax + nll_loss

A fused kernel would:
1. Read logits_proj once
2. Apply softcap in registers
3. Compute log_softmax online (numerically stable)
4. Compute NLL loss and reduce
5. Write only the scalar loss

**BUT with vocab=1024, the logits tensor is small enough to fit in L2 cache (50MB on H100
can hold multiple rows).** The materializing-and-reading-back cost is modest.

**Savings estimate**: With vocab=1024, fused softcap+CE saves ~0.3ms/step.
With vocab=4096+, it would save ~1-2ms/step. **Not worth a custom kernel at vocab=1024.**

modded-nanogpt's fused CE kernel saved 0.9s over the ENTIRE speedrun (~2600 steps) = 0.35ms/step.
But their vocab=50257 is 50× larger than ours.

### Fused RMSNorm + Linear

RMSNorm followed by linear projection happens 3× per block (attn_norm→Q/K/V, mlp_norm→MLP):

```
# Currently (2 separate kernels):
x_norm = F.rms_norm(x, (x.size(-1),))  # Read x, compute norm, write x_norm
q = F.linear(x_norm, w_q.to(x.dtype))  # Read x_norm + w_q, compute matmul, write q
```

Fusing these into one kernel would:
- Read x once, normalize in shared memory
- Immediately feed into the matmul without writing to HBM

**BUT**: The matmul uses Tensor Cores while RMSNorm uses CUDA cores. They have
completely different execution patterns. True fusion is extremely difficult. In
practice, torch.compile places them back-to-back on the same stream, which is
nearly as good for pipelining.

**PyTorch's approach**: `torch.nn.RMSNorm` (the nn.Module version) allows torch.compile
to fuse the norm computation with any subsequent element-wise ops, but not with matmul.

**Savings estimate**: < 0.2ms/step at dim=512. Not worth custom implementation.

### What IS Worth Fusing

The **Muon optimizer's Newton-Schulz** iterations are the best fusion target:

```python
for _ in range(steps):
    A = X @ X.T         # matmul
    B = b * A + c * A @ A  # matmul + element-wise
    X = a * X + B @ X   # matmul + element-wise
```

This is 3 matmuls + 3 element-wise ops per iteration × 5 iterations = 15 matmuls.
Each operates on small matrices (512×512 = 262K elements). These are extremely
memory-bound (OI ≈ 256) and could benefit from a fused Triton kernel that keeps
intermediates in shared memory.

**The `zeropower_via_newtonschulz5` is already `torch.compile`d separately:**
```python
zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
```

This should already fuse the element-wise ops between matmuls. Additional savings
from a custom kernel: ~0.5ms/step.

---

## 6. NCCL OPTIMIZATION FOR DDP

### Current NCCL Configuration
```bash
NCCL_IB_DISABLE=1  # Disable InfiniBand (single-node)
```

### What NCCL Variables Do

| Variable | Purpose | Recommended for 8×H100 |
|----------|---------|----------------------|
| `NCCL_IB_DISABLE=1` | Disable InfiniBand | Correct for single-node |
| `NCCL_ALGO` | Allreduce algorithm | `Ring` for small, `Tree` for large |
| `NCCL_PROTO` | Wire protocol | `LL128` for NVLink, `Simple` for PCIe |
| `NCCL_MIN_NCHANNELS` | Min parallelism | 8-16 |
| `NCCL_MAX_NCHANNELS` | Max parallelism | 32 |
| `NCCL_BUFFSIZE` | Transfer buffer | 16MB for NVLink |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA | Irrelevant for single-node |

### Allreduce Data Volume Analysis

What gets all-reduced per step:
- DDP gradient all-reduce: 17M params × 4 bytes (fp32) = **68 MB**
- Muon updates all-reduce: ~16M params × 2 bytes (bf16) = **32 MB**
- Total: **100 MB per step**

With full NVLink mesh (900 GB/s bidirectional):
- Ring allreduce time: 2 × (N-1)/N × data / BW = 2 × 7/8 × 100MB / 900GB/s = **0.19 ms**

With PCIe (128 GB/s):
- Ring allreduce time: 2 × 7/8 × 100MB / 128GB/s = **1.37 ms**

With pairwise NVLink (4 pairs, cross-pair via PCIe):
- Depends on topology. Worst case: ~2-3ms.

### Recommended NCCL Settings

```bash
# For single-node 8×H100 with NVLink
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring                 # Ring is better for moderate data sizes
export NCCL_PROTO=LL128               # Low-latency 128B protocol for NVLink
export NCCL_MIN_NCHANNELS=16          # More parallel channels
export NCCL_BUFFSIZE=16777216         # 16MB buffer
export NCCL_P2P_LEVEL=NVL            # Use NVLink for peer-to-peer
```

**Expected improvement: 0.5-1.5ms/step (1-3%).** Most of the time is already in compute,
not communication.

### The Real Issue: Muon's All-Reduce

The Muon optimizer does its OWN all-reduce (line 169):
```python
dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
```

This allreduce of ~32MB happens AFTER the DDP gradient allreduce. If we could
overlap them, we'd save 0.5-1ms. But Muon needs gradients from the backward pass
which needs the DDP allreduce. These are strictly sequential.

**Better approach: Reduce DDP overhead by using `gradient_as_bucket_view=True`:**
```python
model = DDP(
    compiled_model,
    device_ids=[local_rank],
    broadcast_buffers=False,
    gradient_as_bucket_view=True,  # Avoids gradient copy
    static_graph=True,             # Enables communication+computation overlap
)
```

`static_graph=True` allows DDP to overlap the allreduce of early buckets with
the backward computation of later layers. This is a **free** 1-2ms saving.

---

## 7. GRADIENT ACCUMULATION ANALYSIS

### Current Setup
```python
grad_accum_steps = 8 // world_size  # = 1 with 8 GPUs
```

With world_size=8, each GPU processes 65,536 tokens per step with zero accumulation.

### Should We Accumulate More?

**No.** Here's why:

1. **Throughput**: With grad_accum=1, we do 1 forward + 1 backward + 1 allreduce per step.
   With grad_accum=2, we do 2 forward + 2 backward + 1 allreduce. The allreduce overhead
   is amortized, BUT the extra forward/backward doubles compute time. Net effect: negative.

2. **The only reason to accumulate**: If the batch doesn't fit in memory. With dim=512
   and 65K tokens, peak memory is ~10.9 GB per GPU (from experiment notes). H100 has 80GB.
   We have massive headroom.

3. **Could we increase total batch size?** The competition uses 524K tokens/step.
   Larger batch = better gradient estimate = fewer steps needed. But with 8 GPUs at
   65K tokens each, we could go to 128K per GPU (1M total) and still fit in memory.
   **This requires changing the training protocol, not just grad accumulation.**

### Alternative: Fewer GPUs + More Accumulation

If Thunder Compute only gives us 4 GPUs:
```python
# 4 GPUs: grad_accum_steps = 8 // 4 = 2
# Each GPU: 65K tokens × 2 accumulations = 130K tokens
# Same total batch as 8 GPUs (524K)
```

This works but step time roughly doubles (2× forward/backward) while saving only
~1ms of allreduce overhead. Net: **~2× slower**.

**Verdict: grad_accum=1 with 8 GPUs is optimal. No change needed.**

---

## SUMMARY: PRIORITIZED RECOMMENDATIONS

### Tier 1: Zero-Risk, Easy Implementation (~5-10ms savings)

| # | Change | Code | Expected Savings | Risk |
|---|--------|------|-----------------|------|
| 1 | `reduce-overhead` compile mode | `torch.compile(..., mode='reduce-overhead')` | **3-5ms** (7-12%) | Low |
| 2 | Enable cuDNN SDP | `enable_cudnn_sdp(True)` | **0-2ms** (0-5%) | Zero |
| 3 | DDP static_graph + bucket_view | See code below | **1-2ms** (2-5%) | Low |
| 4 | VAL_LOSS_EVERY=5000 | Env var | **~500 steps** saved | Zero |
| 5 | MUON_BACKEND_STEPS=3 | Env var | **0.5-1ms** | Low |

### Tier 2: Moderate Risk, Moderate Effort (~3-8ms savings)

| # | Change | Expected Savings | Risk |
|---|--------|-----------------|------|
| 6 | compiled_autograd | **2-4ms** (5-9%) | Moderate |
| 7 | NCCL tuning (LL128, Ring, channels) | **0.5-1.5ms** (1-3%) | Low |
| 8 | Async data prefetch | **0.5-1ms** | Low |

### Tier 3: High Effort, High Reward (~3-5ms savings)

| # | Change | Expected Savings | Risk |
|---|--------|-----------------|------|
| 9 | FP8 for all linears (via torchao) | **3-5ms** (8-12%) | High |
| 10 | FlashAttention-3 | **2-4ms** (5-10%) | High |
| 11 | Backward-optimizer overlap | **2-3ms** (5-7%) | High |

### Total Potential Savings

**Conservative (Tier 1 only): 5-10ms → 33-38ms/step → 15,800-18,200 steps**
**Aggressive (Tier 1+2): 8-16ms → 27-35ms/step → 17,100-22,200 steps**
**Maximum (All tiers): 12-25ms → 18-31ms/step → 19,400-33,300 steps**

Even conservative Tier 1 changes give us **2,000-4,400 more training steps** = 
significant improvement at no convergence risk.

---

## IMPLEMENTATION CODE

### Combined Tier 1 Changes (copy-paste ready)

```python
# ===== In main(), around line 783-791 =====

# CHANGE: Enable cuDNN SDP alongside flash (let PyTorch pick fastest)
enable_cudnn_sdp(True)    # WAS: enable_cudnn_sdp(False)
enable_flash_sdp(True)
enable_mem_efficient_sdp(False)
enable_math_sdp(False)

# ===== Around line 865 =====

# CHANGE: Use reduce-overhead for CUDA graph capture
compiled_model = torch.compile(
    base_model, 
    dynamic=False, 
    fullgraph=True,
    mode='reduce-overhead',  # NEW: CUDA graphs + better fusion
)

# CHANGE: DDP with static_graph and gradient_as_bucket_view
model = DDP(
    compiled_model, 
    device_ids=[local_rank], 
    broadcast_buffers=False,
    gradient_as_bucket_view=True,  # NEW: avoid gradient copy
    static_graph=True,             # NEW: enable compute/comm overlap
) if distributed else compiled_model

# ===== Env vars to set =====
# MUON_BACKEND_STEPS=3
# VAL_LOSS_EVERY=5000
```

### NCCL Environment Variables

```bash
export NCCL_IB_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=LL128
export NCCL_MIN_NCHANNELS=16
export NCCL_BUFFSIZE=16777216
```

---

## KEY INSIGHT

**At dim=512, the model is fundamentally memory-bound. No amount of compute
optimization (FP8, better matmul kernels) will make it compute-bound.** The only
ways to substantially improve throughput are:

1. **Reduce memory traffic**: FP8 (fewer bytes), operator fusion (fewer reads/writes)
2. **Reduce overhead**: CUDA graphs (fewer kernel launches), DDP tuning
3. **Increase arithmetic intensity**: Wider model (dim≥1024 becomes compute-bound)

The **architecture choice matters more than kernel optimization**. Going from
dim=512 to dim=768 BOTH increases quality (more capacity) AND increases MFU
(higher arithmetic intensity). The constraint is the 16MB artifact budget.

This is why **depth recurrence with wider model (e.g., 7×2@672)** is doubly
attractive: it increases both model capacity and hardware utilization. The
challenge is purely in step time vs step count tradeoff.

