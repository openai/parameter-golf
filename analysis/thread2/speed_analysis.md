# Speed Optimization Analysis for Parameter Golf

## Executive Summary

The baseline runs at **43.54ms/step** on 8×H100 achieving 13,780 steps in 10 minutes.
Our best recurrent model (7×2@672) is estimated at **~107ms/step** = ~5,600 steps.
The recurrent model has 0.018 BPB advantage at 2K steps but would get 60% fewer steps,
likely resulting in a NET LOSS of ~0.037 BPB vs baseline at convergence.

**Key finding: Model FLOPs Utilization (MFU) is only 24.2%**, meaning 76% of step time
is overhead (kernel launches, memory operations, communication). This is typical for
small models (17M params) with relatively small batch sizes.

## 1. Where Time Is Spent

### Compute Breakdown (baseline 9@512, per GPU)
| Component | FLOPs | % of Forward | Arithmetic Intensity | Bound |
|-----------|-------|-------------|---------------------|-------|
| Q projection (512→512) | 35 TFLOPS/layer | 7.7% | 255 FLOPs/byte | **MEMORY** |
| K projection (512→256) | 17 TFLOPS/layer | 3.8% | 170 FLOPs/byte | **MEMORY** |
| V projection (512→256) | 17 TFLOPS/layer | 3.8% | 170 FLOPs/byte | **MEMORY** |
| Out projection (512→512) | 35 TFLOPS/layer | 7.7% | 255 FLOPs/byte | **MEMORY** |
| MLP fc (512→1024) | 69 TFLOPS/layer | 15.2% | 340 FLOPs/byte | COMPUTE |
| MLP proj (1024→512) | 69 TFLOPS/layer | 15.2% | 340 FLOPs/byte | COMPUTE |
| Attention QK+SV | 682 TFLOPS/layer | 35.6% | 60 FLOPs/byte | **MEMORY** |
| LM Head (512→1024) | 69 TFLOPS | 2.0% | 340 FLOPs/byte | COMPUTE |

### Time Budget (estimated, 43.54ms total)
| Phase | Estimated Time | Notes |
|-------|---------------|-------|
| Pure compute (at H100 peak) | ~10.5ms | At 989 TFLOPS BF16 |
| Overhead total | ~33ms | 76% of step! |
| - Kernel launch overhead | ~10-15ms | Hundreds of small kernel launches |
| - Memory bandwidth waste | ~8-12ms | Reading/writing intermediates |
| - DDP gradient all-reduce | ~2-3ms | ~68MB gradients over NVLink |
| - Muon optimizer | ~3-5ms | Newton-Schulz + all_reduce |
| - Elementwise ops | ~5-8ms | rms_norm, rotary, scaling, etc. |
| - Data loading | <0.1ms | Negligible |

## 2. Speed Optimization Techniques

### 2.1 torch.compile mode='max-autotune' (5-10% gain, EASY)
**Current**: `torch.compile(base_model, dynamic=False, fullgraph=True)` uses default mode.
**Change**: `torch.compile(base_model, dynamic=False, fullgraph=True, mode='max-autotune')`
**Why**: max-autotune benchmarks multiple kernel implementations for each operation and
selects the fastest. For matmul, it tries different tiling strategies. For elementwise ops,
it tries more fusion patterns.
**Risk**: Compilation time increases from ~60s to ~2-3 min (but warmup steps handle this).
**Code change**:
```python
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True, mode='max-autotune')
```

### 2.2 compiled_autograd (5-10% gain, MODERATE)
**Current**: torch.compile only compiles the forward pass. The backward pass uses eager autograd.
**Change**: Enable compiled_autograd to compile backward too.
**Why**: The backward pass is ~2x the forward. Compiling it enables the same kernel fusion
and launch overhead reduction that torch.compile gives the forward pass.
**Risk**: Experimental feature. May not work with DDP or Muon.
**Code change**:
```python
import torch._dynamo
torch._dynamo.config.compiled_autograd = True
# Or use the context manager:
# with torch._dynamo.compiled_autograd.enable(torch.compile):
```

### 2.3 CUDA Graphs (10-15% gain, HARD)
**Current**: Each kernel is launched individually.
**Change**: Capture the entire fwd+bwd into a CUDA graph and replay it.
**Why**: Eliminates kernel launch overhead (~10-15ms per step).
**Prerequisite**: All shapes must be static (✓ we have this), no CPU-side logic in the
captured region.
**Risk**: Cannot capture NCCL collectives (DDP all-reduce). Must segment the graph
to exclude communication. Muon optimizer's all_reduce also can't be captured.
**Alternative**: `torch.compile` with `mode='reduce-overhead'` automatically uses CUDA
graphs where possible.
**Code change**:
```python
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True, mode='reduce-overhead')
```

### 2.4 FP8 Matmul for MLP (5-8% gain for dim≥672, MODERATE)
**Current**: All matmuls run in BF16 (autocast).
**Change**: Use FP8 (float8_e4m3fn) for the MLP fc and proj matmuls.
**Why**: H100 FP8 tensor cores give ~2x throughput vs BF16. At dim=672, the MLP fc/proj
matmuls are compute-bound (intensity=445 > threshold=295), so FP8 would directly help.
At dim=512, only MLP is compute-bound and the gain is smaller.
**Risk**: FP8 has less numerical precision, may hurt convergence slightly.
**Implementation**: Use `torch.float8_e4m3fn` dtype or `torchao.float8` utilities.
**Note**: For dim=512, the attention projections are memory-bound so FP8 doesn't help
them. Only MLP benefits.

### 2.5 Backward-Optimizer Overlap (3-5% gain, MODERATE)
**Current**: All backward computations finish → then Muon optimizer starts.
**Change**: Register backward hooks that trigger Muon's Newton-Schulz for each parameter
as its gradient becomes available, overlapping NS computation with later layers' backward.
**Why**: The Muon NS for early blocks' parameters could compute while later blocks are
still doing backward.
**Implementation**:
```python
# Register per-parameter backward hooks
def make_hook(param, muon):
    def hook(grad):
        # Start NS for this param's gradient in parallel
        with torch.cuda.stream(muon_stream):
            # Newton-Schulz orthogonalization
            ...
    return hook

for p in matrix_params:
    p.register_hook(make_hook(p, optimizer_muon))
```
**Risk**: Requires careful stream management and synchronization.

### 2.6 Async Data Prefetch (1-3% gain, EASY)
**Current**: Data is loaded synchronously in the training loop via `next_batch()`.
The CPU reads from numpy files, creates tensors, and transfers to GPU with non_blocking=True.
**Change**: Use a background thread to prepare the next batch while the GPU computes.
**Why**: Even though data transfer is fast (0.02ms for 0.5MB), the CPU-side numpy operations
could stall the GPU pipeline briefly.
**Code change**:
```python
import threading
import queue

class AsyncDataLoader:
    def __init__(self, loader, prefetch=2):
        self.loader = loader
        self.queue = queue.Queue(prefetch)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def _worker(self):
        while True:
            batch = self.loader.next_batch(...)
            self.queue.put(batch)
    
    def next_batch(self):
        return self.queue.get()
```

### 2.7 Muon Newton-Schulz Steps: 5 → 3 (1-2% gain, TRIVIAL)
**Current**: 5 iterations of Newton-Schulz orthogonalization.
**Change**: Reduce to 3 iterations.
**Why**: The original Muon paper suggests 5 is generous. 3 iterations may suffice for
convergence quality. Saves ~0.2ms per step.
**Risk**: May slightly hurt optimizer quality. Need to test.
**Code change**: Set `MUON_BACKEND_STEPS=3` in environment or hyperparameters.

### 2.8 DDP Tuning (1-2% gain, EASY)
**Current**: Default DDP settings.
**Change**: `gradient_as_bucket_view=True`, tune `bucket_cap_mb`.
**Why**: `gradient_as_bucket_view` avoids copying gradients into communication buffers.
Tuning bucket size can improve communication scheduling.
**Code change**:
```python
model = DDP(compiled_model, device_ids=[local_rank], 
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=25)  # experiment with 10-50
```

### 2.9 Reduce Validation Overhead
**Current**: Validation runs at step 0, then every 1000 steps (default). Each validation
pass takes several seconds (iterates over all val tokens).
**Change**: Reduce val frequency or subset validation (e.g., 10% of val tokens for
intermediate checks, full validation only at start and end).
**Code change**: `VAL_LOSS_EVERY=2000` or implement fast validation subset.

## 3. Architecture-Level Speed Considerations

### 3.1 The Recurrence Speed Tax
7×2@672 does **2.4x more FLOPs** per forward pass than 9@512:
- 14 effective layers vs 9 layers: 1.56x
- dim=672 vs dim=512: (672/512)² = 1.72x
- Combined: 2.68x theoretical, ~2.45x measured (some overhead is constant)

### 3.2 Optimal Config Within 16MB Budget
These configs FIT in 16MB and have interesting speed/quality tradeoffs:

| Config | Params | Step (ms) | Steps/10min | Eff. Layers |
|--------|--------|-----------|-------------|-------------|
| 9@512 (baseline) | 17.06M | 43.5 | 13,780 | 9 |
| 11@464 | 17.08M | 43.7 | 13,728 | 11 |
| 9@496 | 16.03M | 40.9 | 14,683 | 9 |
| 10@480 | 16.64M | 42.5 | 14,111 | 10 |
| 7×2@576 | 16.86M | 85.7 | 6,999 | 14 |
| 5×2@672 | 16.51M | 83.3 | 7,199 | 10 |

**Key insight**: 11@464 has the SAME step time as baseline but 2 more layers (11 vs 9).
It has nearly identical param count (17.08M vs 17.06M). This could be a free improvement
IF the extra depth compensates for the slightly narrower model.

### 3.3 The Speed Paradox with Recurrence
For recurrence to win, it needs to provide enough per-step quality improvement to
overcome getting 40-60% fewer total steps. Based on our experiments:
- 7×2@672 at 2K steps: 1.2797 (0.018 BPB better than baseline at 2K)
- Projected at 5.6K steps: ~1.254 BPB
- Baseline at 13.8K steps: 1.2172 BPB
- **Recurrence LOSES by ~0.037 BPB** despite per-step advantage

To make recurrence viable, we need to either:
1. Speed up by 33%+ (bringing 107ms → 72ms, getting 8300+ steps)
2. Or accept recurrence is a dead end and optimize the baseline instead

## 4. Recommended Actions

### Priority 1: Quick Wins (implement immediately)
1. `mode='reduce-overhead'` in torch.compile (CUDA graphs + autotune)
2. `gradient_as_bucket_view=True` in DDP
3. Muon NS steps 5 → 3
4. Validate less often (VAL_LOSS_EVERY=2000+)

### Priority 2: Medium Effort
5. compiled_autograd for backward
6. Async data prefetch
7. FP8 for MLP matmuls (only if using dim≥672)

### Priority 3: Test Alternative Architectures
8. **11@464** — same speed as baseline but 2 more layers (needs artifact size check)
9. **10@480** — 10 layers at similar speed

### Priority 4: Only If Recurrence Wins Quality Tests
10. Backward-optimizer overlap
11. Custom Triton fused kernels

## 5. Critical Realization

**The biggest speed win may not be per-step optimization — it may be choosing the right
architecture.** An 11-layer model at dim=464 would get the same ~13,700 steps as baseline
but with 22% more depth. If each additional layer provides diminishing but nonzero value,
this is a FREE improvement worth testing before investing in complex speed optimizations.

Similarly, a 10-layer model at dim=480 gets 14,111 steps (2.4% more than baseline) with
11% more depth. Both of these fit within 16MB.
