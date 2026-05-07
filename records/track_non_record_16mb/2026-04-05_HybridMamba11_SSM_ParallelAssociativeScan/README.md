# HybridMamba-11: First SSM Submission with torch.compile-Compatible Parallel Scan

**Non-Record Submission (Research Contribution + Credit Request)**
**Author:** Jesus Perez Bazarot ([@PersusUS](https://github.com/PersusUS))
**Duration:** ~1 week of development, 1xH100 preliminary runs
**Current best:** 2.12 bpb (1xH100, 312 steps, no fullgraph compile) — not competitive yet, see below for why and the path forward.

---

## Summary

This is the first SSM-based submission to Parameter Golf, targeting the unchecked "State-space models" checkbox in the README. The architecture is a hybrid Mamba SSM + Transformer with a novel parallel associative scan that makes Mamba fully compatible with `torch.compile(fullgraph=True)`.

**The core contribution is not the BPB (which is not yet competitive) but the engineering solution to the SSM compilation barrier** — the problem that killed every prior SSM attempt in this competition (see Issue #140/Hymba, PR #831 analysis).

---

## Architecture: HybridMamba-11

11 layers alternating Mamba SSM and Transformer blocks with U-Net skip connections:

```
Layer 0:  MambaBlock (SSM)        <- encoder
Layer 1:  TransformerBlock        <- encoder
Layer 2:  MambaBlock (SSM)        <- encoder
Layer 3:  TransformerBlock        <- encoder
Layer 4:  MambaBlock (SSM)        <- encoder
--- encoder/decoder split ---
Layer 5:  TransformerBlock        <- decoder (+ skip from layer 4)
Layer 6:  MambaBlock (SSM)        <- decoder (+ skip from layer 3)
Layer 7:  TransformerBlock        <- decoder (+ skip from layer 2)
Layer 8:  MambaBlock (SSM)        <- decoder (+ skip from layer 1)
Layer 9:  TransformerBlock        <- decoder (+ skip from layer 0)
Layer 10: TransformerBlock        <- decoder
```

- 5 Mamba SSM layers + 6 Transformer layers
- Parameter banks only for Transformer layers (Mamba layers have independent params)
- 31.8M parameters, 13.6MB compressed artifact (under 16MB cap)
- All SOTA techniques preserved: LeakyReLU(0.5)^2, GQA, XSA, EMA/SWA, BigramHash, SmearGate, int6 mixed quantization + LZMA

---

## The Problem: SSMs Can't Compile

Every SSM attempt in Parameter Golf has hit the same wall: `torch.compile(fullgraph=True)` gives ~10x training speedup, but Mamba's selective scan is a sequential loop (`for i in range(L)`) that torch.dynamo can't trace. Without fullgraph compile, you get ~300 steps in 600s instead of ~20,000. Game over.

Prior approaches:
- **mamba_ssm CUDA kernels**: Fast but pybind11 extensions break dynamo tracing
- **Pure PyTorch sequential loop**: Traceable but inherently sequential — can't parallelize
- **Per-block compile** (my initial fix): Compile Transformer blocks individually, leave Mamba uncompiled. Partial speedup but ~8x slower than SOTA per step.

---

## The Solution: Parallel Associative Scan

The SSM recurrence `h[t] = A[t] * h[t-1] + B[t] * u[t]` is a **linear recurrence** — parallelizable via a Hillis-Steele-style prefix scan (Hillis & Steele, 1986). I implemented a Hillis-Steele parallel associative scan in pure PyTorch:

```python
def parallel_associative_scan(gates, values):
    """O(L log L) work, O(log L) depth. torch.compile(fullgraph=True) compatible."""
    B, L = gates.shape[:2]
    a, b = gates, values
    for d in range(int(math.ceil(math.log2(max(L, 2))))):
        stride = 2 ** d
        if stride >= L: break
        new_b = torch.cat([b[:, :stride], a[:, stride:] * b[:, :L-stride] + b[:, stride:]], dim=1)
        new_a = torch.cat([a[:, :stride], a[:, stride:] * a[:, :L-stride]], dim=1)
        a, b = new_a, new_b
    return b
```

### Validation Results (RTX 4050)

| Test | Result |
|------|--------|
| Correctness (all seq lengths 1-2048) | max_err < 1e-6 |
| Gradient match (sequential vs parallel) | max_err < 2.3e-5 |
| FP16 numerical stability | No NaN/Inf |
| Throughput vs sequential loop | **48x speedup** |
| torch.compile(fullgraph=True) dynamo trace | **SUCCESS** (forward + backward) |
| Full MambaSSM module integration | forward + backward OK |

The Triton/inductor backend requires Linux (not available on Windows dev machine). **H100 validation is the next step** — the reason for this credit request.

---

## Preliminary Results (1xH100, 600s)

| Metric | Value |
|--------|-------|
| Parameters | 31.8M |
| Compressed artifact | 13.6MB |
| val_bpb (pre-EMA, step 312) | 2.12 |
| val_bpb (int6 quantized) | 2.80 |
| Steps completed | 312 / 20,000 |
| Step time | 1.93s (1xH100, per-block compile, no fullgraph) |

**Why the BPB is not competitive:** Without fullgraph compile, only 312 steps complete in 600s. The SOTA completes ~20,000 steps. With parallel scan enabling fullgraph compile, projected step time drops to <0.1s, enabling 15,000-20,000 steps — enough for competitive BPB.

---

## Engineering Challenges Solved

1. **torch.compile incompatibility** — mamba_ssm CUDA kernels not traceable. Solved with parallel associative scan (pure PyTorch, O(log L) depth).
2. **dtype mismatch** — `restore_low_dim_params_to_fp32` broke causal_conv1d kernels. Fixed by excluding SSM params from FP32 restoration.
3. **Dead parameter elimination** — Banks originally allocated for all 11 layers. Restructured for 6 Transformer layers only (-12.5M params, -30%).
4. **State dict compatibility** — `_orig_mod.` prefix from torch.compile wrappers broke eval model loading. Fixed with prefix stripping.

---

## What's Needed (Credit Request Justification)

1. **H100 Triton/inductor validation** — Verify fullgraph compile produces correct kernels and measure actual step time ($5-10)
2. **8xH100 DDP competition runs** — Full 600s runs at competition scale ($50-80)
3. **Architecture search** — Layer ratios, d_state tuning, sequence length experiments ($20-30)

---

## Reproducing

```bash
# Prerequisites: PyTorch 2.x with CUDA, sentencepiece, flash-attn
# Download FineWeb data first (see repo README)

# Single GPU smoke test:
python train_gpt.py

# Competition run (8xH100):
MAX_WALLCLOCK_SECONDS=600 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Force CUDA mamba_ssm kernels (breaks fullgraph compile):
MAMBA_USE_CUDA=1 python train_gpt.py
```

---

## Acknowledgments

- The Parameter Golf community for the incredible open research culture
- PR #831 for the throughput-quantization co-optimization analysis that clarified why SSMs fail
- Issue #140 (Hymba) for demonstrating the SSM potential and the compilation barrier
- Hillis & Steele (1986) for the parallel prefix scan algorithm that makes this possible
