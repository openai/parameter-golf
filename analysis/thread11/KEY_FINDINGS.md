# Thread 11: KEY FINDINGS — H100 Kernel Optimization

## The #1 Finding: 71.7% of Step Time is Overhead

```
Step time breakdown (43.5ms total):
├── Forward matmuls:     2.41 ms  ( 5.5%)
├── Forward attention:   0.63 ms  ( 1.4%)
├── Backward pass:       6.08 ms  (14.0%)
├── DDP communication:   0.19 ms  ( 0.4%)
├── Muon optimizer:      3.00 ms  ( 6.9%)  [estimated]
└── OTHER OVERHEAD:     31.19 ms  (71.7%)  ← THE BOTTLENECK
    ├── Kernel launch overhead: ~10-15ms (hundreds of small kernels)
    ├── Element-wise ops:       ~8-12ms  (norms, scales, RoPE, activation)
    └── CUDA sync/scheduling:   ~5-8ms
```

**We spend 5.26ms doing useful compute and 38.24ms waiting.** MFU = 12.1%.

## Ranked Recommendations

### 1. `reduce-overhead` in torch.compile — Expected: 5-10ms savings (12-23%)
The single biggest win. CUDA graphs eliminate ~10ms of kernel launch overhead.
```python
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True, mode='reduce-overhead')
```
**CAUTION**: May conflict with DDP. Test carefully. At minimum, forward pass gets graphed.

### 2. DDP `static_graph=True` + `gradient_as_bucket_view=True` — Expected: 1-2ms (2-5%)
Free optimization. Overlaps communication with backward computation.
```python
model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False,
            gradient_as_bucket_view=True, static_graph=True)
```

### 3. Enable cuDNN SDP — Expected: 0-2ms (0-5%)
Zero risk. Let PyTorch benchmark cuDNN vs Flash backends and pick the winner.
```python
enable_cudnn_sdp(True)  # Was False
```

### 4. `MUON_BACKEND_STEPS=3` — Expected: 0.5-1ms
Reduce Newton-Schulz iterations from 5 to 3. Minimal convergence impact per literature.

### 5. `VAL_LOSS_EVERY=5000` — Expected: ~500 extra training steps
Saves 20+ seconds of validation overhead during training.

### 6. FP8 matmul (all linears) — Expected: 3-5ms (8-12%)
Higher risk. Every matmul is memory-bound (OI=170-340 vs crossover=591).
FP8 reduces memory traffic by 25-33% → 1.2-1.5× per matmul.
Use `torchao.float8` for compile compatibility.

### 7. FlashAttention-3 — Expected: 2-4ms (5-10%)
Requires separate installation. Uses H100 async Tensor Cores.
May break torch.compile fullgraph unless registered as custom op.

## FP8 Verdict: Modest at dim=512, Better at dim=672+

All operations at dim=512 are MEMORY-BOUND (OI: 170-340, crossover: 591).
FP8's 2× compute doesn't help. FP8's 1-byte reads (vs 2-byte BF16) saves
25-33% memory traffic → only 1.2-1.5× speedup per matmul.

At dim=672, MLP operations reach OI=448 — still memory-bound but closer.
At dim=896, MLP becomes compute-bound and FP8 truly doubles throughput.

**For dim=512 baseline: FP8 saves ~3ms (8% total). Marginal.**
**For dim=672 recurrent: FP8 saves ~5-7ms (5-7% of ~100ms). More valuable.**

## The Architecture Insight

The most impactful optimization is NOT kernel-level — it's **choosing dims that
are compute-bound**. At dim=512, we waste 88% of H100 peak TFLOPS.

| Dim | MLP OI | Q/O proj OI | Bound | MFU (est.) |
|-----|--------|-------------|-------|------------|
| 512 | 341 | 256 | MEMORY | 12% |
| 768 | 512 | 384 | MEMORY | 23% |
| 1024 | 683 | 512 | MIXED | 38% |
| 1536 | 1024 | 768 | COMPUTE | 55% |

But we CAN'T go wider (16MB artifact limit). So we're stuck with low MFU.

## Combined Impact Estimate

| Scenario | Changes | Step Time | Steps/10min | vs Baseline |
|----------|---------|-----------|-------------|-------------|
| Baseline | None | 43.5ms | 13,780 | — |
| Conservative | #2,#3,#4,#5 | 40.5ms | 14,815 | +7.5% |
| Moderate | + #1 | 35.5ms | 16,901 | +22.6% |
| Aggressive | + #6,#7 | 30.5ms | 19,672 | +42.7% |

**Even conservative changes give ~1,000 extra steps. Moderate gives ~3,000.**
**Each extra step at the end (during warmdown) is worth ~0.00005 BPB.**
**3,000 extra steps ≈ 0.01-0.02 BPB improvement from more training alone.**
