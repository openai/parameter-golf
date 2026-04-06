# EXP-002: Hadamard Pre-Rotation + GPTQ int6

## Metadata
- **Date**: 2026-04-06 (planned)
- **Branch**: exp/hadamard-gptq-int6
- **Parent**: exp/reproduce-sota
- **Priority**: P0
- **Estimated runs**: 1 dev + 1 full + 3 seed = 5 runs
- **Estimated cost**: ~$17
- **Papers**:
  - TurboQuant (arXiv:2504.19874, ICLR 2026) - rotation + scalar quant for KV cache
  - PolarQuant weight variant (arXiv:2603.29078) - Hadamard rotation for weight quant
  - QuIP# (arXiv:2402.04396) - Hadamard incoherence for LLM quantization

## Hypothesis
Applying a Walsh-Hadamard rotation to each weight matrix before GPTQ quantization will
reduce quantization error at int6 because:
1. LLM weight matrices have outlier columns that cause disproportionate quantization error
2. The Hadamard transform spreads these outliers uniformly across all columns
3. After rotation, the weight distribution is more Gaussian/uniform, making per-column
   quantization more effective
4. GPTQ's Cholesky error compensation works better on a smoother weight landscape
5. QuIP# demonstrated that rotation alone can improve quantization by ~1 perplexity point

The rotation is applied once post-training (before GPTQ) and the inverse rotation is
applied during dequantization. No additional parameters are needed.

## Null Hypothesis
The GPTQ algorithm already compensates for outlier columns via its column reordering
and error feedback. The Hadamard rotation adds overhead (inverse rotation during eval)
without improving quantization quality at int6. Or: int6 is already high enough bits
that outliers aren't the bottleneck.

## Control Variables (what stays the same)
- Full model architecture (11L, XSA-all, BigramHash 3072x112, etc.)
- Training loop, optimizer, weight averaging (all identical)
- AR self-gen calibration (64 seqs x 2048 tokens)
- LZMA preset=9, selective pruning
- Same seeds, same eval protocol

## Independent Variable (what changes)
In the GPTQ quantization pipeline (lines 1171-1224 of SOTA script):
- BEFORE quantizing each weight matrix W:
  - Generate a random sign vector s (from fixed seed)
  - Apply randomized Hadamard transform: W_rot = W @ diag(s) @ H_n / sqrt(n)
  - H_n is the Walsh-Hadamard matrix (computed in O(n log n), never stored)
- Quantize W_rot using existing GPTQ pipeline
- DURING dequantization:
  - After dequantizing W_rot_q, apply inverse: W_q = W_rot_q @ H_n^T @ diag(s) / sqrt(n)
  - (Hadamard is its own inverse up to scaling)

Also must transform the Hessian: if H = X^T X for the original weights,
the rotated Hessian is H_rot = R^T H R where R is the rotation matrix.

## Implementation Details
The Walsh-Hadamard transform can be applied in-place in O(n log n):
```python
def hadamard_transform(x):
    """In-place Walsh-Hadamard transform along last dimension."""
    n = x.shape[-1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a, b = x[..., j], x[..., j + h]
                x[..., j] = a + b
                x[..., j + h] = a - b
        h *= 2
    x /= n ** 0.5
    return x
```

Or use torch FFT-based implementation for GPU efficiency.
Requires weight dimensions to be powers of 2 (512 = 2^9, OK).

## Success Criteria
- BPB < 1.1147 (any improvement)
- Artifact size <= 15.95 MB (rotation adds no parameters, only eval overhead)
- Eval time increase < 30s (inverse Hadamard is fast)

## Abort Criteria
- BPB > 1.1160 on dev run (worse than baseline -> rotation hurts GPTQ)
- Eval time > 10 minutes (exceeds eval budget)
- Dequant numerical instability (NaN/Inf)

## Run Plan
1. DEV RUN: seed=314, 8xH100
   - Goal: verify Hadamard + GPTQ pipeline produces valid weights, rough BPB
2. DECISION GATE: BPB < 1.1160 AND no numerical issues
3. FULL RUN: seed=314
4. DECISION GATE: BPB < reproduced baseline (1.1147)
5. SEED RUNS: seeds 314, 42, 999

## Implementation Plan
1. Copy SOTA train_gpt.py to new branch
2. Add hadamard_transform() utility function
3. Modify quantize_int6_gptq() to rotate W before quantization
4. Modify dequantize_mixed_int6() to apply inverse rotation after dequant
5. Transform Hessians to match rotated weight space
6. Add HADAMARD_ROTATE env var toggle (default 1)
7. Test compilation locally
8. Run on 8xH100

## Risks
- Hadamard requires dim to be power of 2. model_dim=512 is fine. MLP hidden=1536
  is NOT a power of 2 -> need to pad to 2048 or handle non-power-of-2 case.
  QuIP# handles this with zero-padding. Check if this adds too much overhead.
- The inverse rotation during eval adds latency to every forward pass.
  Need to benchmark: for 512-dim, Hadamard is ~4.5K FLOPs per matrix, negligible.

## Results
[Fill in after running]

## Post-Mortem
[Fill in after running]
