---
name: triton-persistent-warp-matmul
description: Teach an AI agent to implement persistent, warp-specialized matmul kernels in Triton using TMA and producer/consumer warps.
---

# Persistent & Warp-Specialized Matmul Kernels in Triton

> **Targets:** Triton >= 3.0; TMA/warp specialization requires SM90+ (Hopper)

Overview
This skill teaches how to implement a persistent GEMM in Triton where fewer thread blocks than output tiles are launched and each block iterates over multiple tiles. It covers tile scheduling (linear tile_id → 2D via `//` and `%`), persistent loop strides, TMA/device descriptors, producer/consumer warp roles, and epilogue subtiling for memory efficiency.

Step-by-step / Key principles
1. Partitioning and constants:
   - Define tile sizes BLOCK_M × BLOCK_N and inner block BLOCK_K.
   - num_tiles = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N). Use cdiv(x,y) = (x+y-1)//y.
2. Persistent scheduling:
   - Launch num_blocks < num_tiles. Each block computes:
     for tile_id in range(start_tile + block_id, num_tiles, num_blocks)
   - Convert linear tile_id to 2D: m_block = tile_id // num_tiles_n; n_block = tile_id % num_tiles_n. (Note: Python `divmod` is not supported in Triton JIT — always use `//` and `%`.)
3. Warp specialization:
   - Split warps into producers (async TMA loads or tl.async_copy into shared memory) and consumers (wait on barrier, compute tl.dot).
   - Producers write tiles to sA/sB, then tl.barrier(); consumers perform tl.dot using shared tiles.
4. TMA / async loads:
   - On SM90+, create device descriptors: desc = tl.make_tensor_descriptor(ptr, shape, strides, block_shape) and use tl.tma_load / tl.tma_store.
5. Epilogue and subtile:
   - Write output in subtile chunks to reduce shared memory and register pressure.
6. Numerical and synchronization:
   - Use fp32 accumulators for mixed precision and careful barrier placement between producer/consumer groups.

Practical examples

### Persistent matmul with grouped ordering (from KernelAgent/Meta)

Launch only `NUM_SMS` blocks, each looping over tiles with `tl.range(..., flatten=True)` for software pipelining:

```python
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

@triton.jit
def matmul_persistent(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Duplicate tile counter for epilogue (workaround for pipelining bug)
    tile_id_c = start_pid - NUM_SMS

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Compiler hints for aligned accesses
        offs_m = tl.where(offs_m < M, offs_m, 0)
        offs_n = tl.where(offs_n < N, offs_n, 0)
        offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_K + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                        mask=offs_k[None, :] < K - ki * BLOCK_K, other=0.0)
            b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                        mask=offs_k[:, None] < K - ki * BLOCK_K, other=0.0)
            acc = tl.dot(a, b, acc)

        # Epilogue: recompute pid for output (separate counter avoids pipelining issue)
        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
                 acc.to(tl.float16), mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
```

Launcher — launch exactly `NUM_SMS` blocks:
```python
NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"])),)
matmul_persistent[grid](a, b, c, M, N, K, ..., NUM_SMS=NUM_SMS)
```

**Key patterns:**
- `tl.range(start, end, stride, flatten=True)` enables software pipelining across tile iterations
- `tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)` hints the compiler for vectorized loads
- `tl.where(offs < dim, offs, 0)` replaces masking with clamping for aligned access patterns
- Separate `tile_id_c` counter for epilogue avoids values crossing prologue/epilogue boundary

### TMA descriptor pattern (SM90+ / Hopper)

Use `TensorDescriptor` for hardware-accelerated memory transfers:
```python
from triton.tools.tensor_descriptor import TensorDescriptor

# Launcher — create descriptors with dummy block_shape (autotune fills real values)
y_dim = B * H * SEQ_LEN
desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[1, 1])

# Kernel — load via descriptor (no pointer arithmetic needed)
@triton.jit
def _kernel(desc_q, desc_k, desc_v, desc_o, ...):
    desc_q = tl.make_tensor_descriptor(desc_q, shape=[y_dim, HEAD_DIM],
                                        strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
    q = desc_q.load([offset_y, 0])            # hardware TMA load
    desc_o.store([offset_y, 0], out.to(dtype)) # hardware TMA store
```

Best practices & pitfalls
- **Persistent vs standard:** Persistent kernels win when kernel launch overhead is significant (many small tiles) or when overlapping loads and compute improves utilization. For large single-tile problems, standard grids may be simpler and equally fast.
- **`NUM_SMS`:** Always query `torch.cuda.get_device_properties("cuda").multi_processor_count` — don't hardcode.
- Tune BLOCK_M/BLOCK_N to balance shared memory, registers, and TMA granularity.
- Ensure correct alignment and `block_shape` when creating TMA descriptors.
- Carefully design producer/consumer warp split to avoid idle warps.
- Profile with Triton Proton and compare against cuBLAS.