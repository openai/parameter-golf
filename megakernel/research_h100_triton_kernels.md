# H100 Triton Kernel Optimization Research
# Compiled: 2026-05-04 | Parameter Golf Sprint

## TL;DR — What Actually Works on H100

| Technique | Speedup | Source |
|-----------|---------|--------|
| Persistent kernel (grid=132 SMs) | Eliminates wave quantization | PyTorch blog |
| Grouped tile ordering (GROUP_SIZE_M=8) | **1.33x, +60% L2 hit** | PyTorch MoE blog |
| TMA (Tensor Memory Accelerator) | Frees SM resources | NVIDIA / H100 worklog |
| BM=128, BN=256, BK=64 tiles | **631 TFLOPs** (H100) | H100 GEMM worklog |
| Warp specialization (1P+2C warpgroups) | 631→704+ TFLOPs | H100 GEMM worklog |
| PTX barriers vs CUDA barriers | **10% boost** | H100 GEMM worklog |
| Thread block clusters (2-SM) | TMA multicast | H100 GEMM worklog |
| Hilbert curve tile ordering | +1% | H100 GEMM worklog |
| Fused RMSNorm (Liger style) | **6x over PyTorch** | Liger-Kernel paper |
| autoWS warp specialization | 1.5-2x over stock Triton | PyTorch autoWS blog |

---

## Section 1: Persistent Kernels

### Why Persistent Matters
- Non-persistent: M/BLOCK_M × N/BLOCK_N kernel launches → wave quantization wastes SMs
- Persistent: grid = (132, 1, 1) — exactly one program per SM, all stay alive
- Each program loops: `for tile_id in tl.range(start_pid, num_tiles, NUM_SMS)`

### Implementation Pattern (from PyTorch grouped GEMM blog)
```python
NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count  # 132 on H100

grid = (NUM_SMS, 1, 1)

@triton.jit
def persistent_gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_tiles_m = tl.cdiv(M, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_tiles_m * num_tiles_n
    
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n
        # ... process tile (pid_m, pid_n)
```

### Grouped Tile Ordering (Critical — 1.33x speedup)
```python
# WRONG (row-major): C(0,0) → C(0,1) → C(0,2) → C(1,0)  — cold A matrix every time
# RIGHT (grouped):  C(0,0) → C(1,0) → C(2,0) → C(0,1)  — keep A rows in L2 cache

GROUP_SIZE_M = 8  # tested: 8 works well for H100

def get_grouped_pid(tile_id, num_tiles_m, num_tiles_n, GROUP_SIZE_M):
    num_pid_in_group = GROUP_SIZE_M * num_tiles_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_tiles_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n
```

**Result (PyTorch MoE blog, H100):**
- Baseline (linear): 1.0x
- Grouped tiling: **1.33x speedup**, +60% L2 cache hit rate

---

## Section 2: Optimal Block Sizes for H100 BF16 GEMM

### From H100 GEMM Worklog (Pranjal, cudaforfun.substack.com)
Competition token shape: M=73728, K=512, N=1536

| Config | TFLOPs | Notes |
|--------|--------|-------|
| BM=128, BN=128, BK=64 | 423 | Basic tensor cores |
| BM=128, BN=256, BK=64 | **631** | 2 consumer warpgroups — sweet spot |
| + PTX barriers | 704 | 10% from barrier switch |
| + Thread block clusters | 734 | TMA multicast |
| + Async stores | 758 | TMA output stores |
| + Hilbert ordering | 764 | +1% cache |

**For our competition shape (M=73728, K=512, N=1536):**
- Recommended: `BLOCK_M=128, BLOCK_N=128 or 256, BLOCK_K=64`
- `num_stages=3 or 4` (software pipelining for HBM latency hiding)
- `num_warps=8` (128 threads = 1 warpgroup)

### From Official Triton Persistent Matmul Tutorial
```python
# Tested configs that work on H100:
configs = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=4, num_warps=8),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32},  num_stages=2, num_warps=4),
]
```

---

## Section 3: TMA (Tensor Memory Accelerator)

### What It Is
- H100-exclusive hardware unit for 2D tile transfers
- Single thread issues async load/store (frees 127 other threads for compute)
- Automatic swizzle mode eliminates bank conflicts
- Requires `triton.tools.tensor_descriptor.TensorDescriptor`

### Triton TMA Usage
```python
from triton.tools.tensor_descriptor import TensorDescriptor

# Creating a TMA descriptor
desc_a = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
desc_b = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])

@triton.jit
def kernel_with_tma(desc_a, desc_b, ...):
    # Load using TMA — ONE thread issues, hardware manages
    a_tile = tl._experimental_descriptor_load(desc_a, [pid_m * BLOCK_M, k * BLOCK_K],
                                               [BLOCK_M, BLOCK_K], tl.bfloat16)
```

### Performance Impact
- Eliminating one intermediate tensor write (e.g., normed_x ~75MB): saves ~0.023ms per layer
- 22 layers × fwd+bwd = ~1ms per step at H100 HBM3 bandwidth

---

## Section 4: Warp Specialization on H100

### Architecture
- H100 has 4 warp schedulers per SM
- Warp specialization assigns different warp groups to async roles:
  - **Producer warpgroup** (128 threads): manages TMA loads, moves data
  - **Consumer warpgroups** (128 threads × N): execute WGMMA, compute

### Config Pattern
```python
# From PyTorch autoWS blog
@triton.jit
def specialized_kernel(...):
    for k_tile in tl.range(lo, hi, BLOCK_K, warp_specialize=True):
        # TMA loads happen in producer; WGMMA in consumers
        ...
```

### From WGMMA Worklog (1 producer + 2 consumers = 384 threads)
```
1 producer warpgroup  = 128 threads, manages async TMA
2 consumer warpgroups = 256 threads, each runs m64n128k16 WGMMA
QSIZE = 3-5 (circular buffer depth)
Result: 631 TFLOPs (up from 423 at 128+128 tiles)
```

### Performance
- autoWS on B200: 1.5-2x over stock Triton
- H100 warp specialization (manual): ~1.33x-1.5x
- Enables full overlap of TMA loads with WGMMA compute

---

## Section 5: RMSNorm Fusion Techniques

### The Reordering Trick (Our Kernel's Core)
```
Standard:  normed = x / rms(x)    [read x, compute rms, write normed]
           out = normed @ W        [read normed, matmul, write out]

Fused:     during GEMM k-loop: accumulate (x_tile @ W_col_tile) AND sum(x_tile²)
           after k-loop: scale accumulator by inv_rms = scale / sqrt(sum_sq/K + eps)
           ELIMINATES the write+read of normed_x (~75MB per GPU)
```

### Liger-Kernel RMSNorm (reference implementation)
- Fuses norm + scale in single Triton kernel
- Caches `rms` values for backward pass
- **6x faster** than PyTorch's separate ops on A100
- Source: https://github.com/linkedin/Liger-Kernel

### Two-Pass Approach (alternative)
```python
# Pass 1: compute inv_rms only (fast, memory-light)
@triton.jit
def compute_inv_rms(x_ptr, inv_rms_ptr, M, K, eps, scale, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid * K + tl.arange(0, BLOCK_K))
    sum_sq = tl.sum(x.float() * x.float())
    inv_rms = scale / tl.sqrt(sum_sq / K + eps)
    tl.store(inv_rms_ptr + pid, inv_rms)

# Pass 2: standard GEMM but scale A tiles by inv_rms[row]
# inv_rms loaded once per BLOCK_M rows, applied inline to accumulator
```

### Unified QKV (Single x-read for Q+K+V)
```python
# Standard: 3 separate RMSNorm → 3 separate linear → 3 reads of x (or normed_x)
# Unified:  Read x ONCE in outer loop, compute sum_sq ONCE, reuse inv_rms for Q+K+V
# Register pressure higher but memory savings ~3x normed_x write
```

---

## Section 6: WGMMA Instruction Details

### Instruction Spec
```
wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
                               ^  ^  ^
                               m  n  k  (matrix dims per warpgroup)
```
- **65,536 MACs per instruction** (m64×n64×k16 = 65536)
- 128 threads cooperate (1 warpgroup = 4 warps)
- Inputs from A-descriptor (shared mem) + B-descriptor (shared/global)
- Output: FP32 accumulators in registers (1024 registers for C, 128 per thread)

### For our K=512, N=1536 shape
- K=512 → 32 k-iterations at k16
- N=1536 → 24 n-columns at n64
- M=73728 → 1152 m-rows at m64
- Best approach: BM=128 = 2× m64 warpgroups, BN=128 = 2× n64

---

## Section 7: Thread Block Clusters

### What It Is
- H100-only: group of 2-8 SMs that share L2 locality
- `__cluster_dims__(2, 1, 1)` → 2 SMs per cluster
- TMA multicast: load one tile, broadcast to all cluster members

### In Triton
```python
@triton.jit
def cluster_kernel(...):
    # Grid = (num_tiles // cluster_size, cluster_size, 1)
    # Programs in same cluster share data via distributed shared memory
```

### When to Use
- When multiple tiles need same A or B data (batch GEMM, attention)
- Not needed for skinny matrices (small M or N)
- For our shape M=73728: potentially useful for N dimension

---

## Section 8: Autotune Config Space for Our Kernels

### Recommended Config Grid (run on H100)
```python
@triton.autotune(
    configs=[
        # Standard configs
        triton.Config({"BM": 64,  "BN": 64,  "BK": 32}, num_warps=4, num_stages=2),
        triton.Config({"BM": 64,  "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64,  "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=4),
        # H100 sweet spot
        triton.Config({"BM": 128, "BN": 256, "BK": 64}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 256, "BK": 64}, num_warps=8, num_stages=4),
        # Large tiles
        triton.Config({"BM": 256, "BN": 128, "BK": 64}, num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 256, "BK": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
```

### Key Finding: num_stages on H100
- H100 HBM3 bandwidth: 3.35 TB/s (vs A100 2.0 TB/s)
- Higher bandwidth → lower latency to hide → num_stages=3 often optimal (vs 4+ on A100)
- For small K (K=32,64): num_stages=2 enough
- For K=512 (our case): num_stages=3-4

---

## Section 9: RMSNorm+QKV Fusion Correctness Notes

### Backward Pass Dtype Rules
- ALL gradient matmuls must be in FP32 to avoid NaN/overflow:
  ```python
  # CORRECT:
  dw_q = dq_2d.float().T @ x_normed.float()  # both sides .float()
  dx   = d_xn.float() @ w.float()
  ```
- inv_rms must be stored as FP32 (not BF16) for backward correctness

### COMPUTE_RMS Flag Pattern
```python
COMPUTE_RMS: tl.constexpr  # True for Q pass, False for K/V (loads cached inv_rms)
if COMPUTE_RMS:
    inv_rms = scale / tl.sqrt(sum_sq / K + eps)
    tl.store(inv_rms_ptr + offs_m, inv_rms, mask=mask_m)
else:
    inv_rms = tl.load(inv_rms_ptr + offs_m, mask=mask_m)
```

---

## Key Sources

- [H100 GEMM Worklog (cudaforfun)](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) — block sizes, warpgroup configs, TFLOPs
- [PyTorch Persistent Cache-Aware GEMM Blog](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/) — 1.33x tile ordering
- [Triton Persistent Matmul Tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) — NUM_SMS pattern
- [Warp Specialization in Triton](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/) — autoWS, warp_specialize=True
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — fused RMSNorm 6x speedup
- [Anatomy of a Triton Attention Kernel](https://arxiv.org/html/2511.11581v1) — attention tiling, H100 warp spec future work
- [Hamza H100 GEMM Worklog](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/) — WGMMA specs, vectorized tiling
