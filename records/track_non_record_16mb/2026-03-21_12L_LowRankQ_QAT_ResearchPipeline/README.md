# 12-Layer Low-Rank Q + QAT: A Cross-Disciplinary Research Pipeline

**Non-record submission** — developed on 1xH100, awaiting 8xH100 for official scoring.

## Results

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb (1xH100, 7900 steps) | **1.2035** |
| Projected post-quant (int6 + sliding window s64) | **~1.19** |
| Architecture | 12L, 512d, MLP 3x, Low-Rank Q (r=128) |
| Params | ~20.9M |
| Artifact | ~15.2MB (uniform int6 + zstd-22) |
| Projected 8xH100 step time | ~77ms → ~7800 steps in 10min |

## Approach

We started from the current SOTA techniques (int6 quantization, MLP 3x, SmearGate, BigramHash, Muon WD, overtone spectral init, sliding window eval) and asked: **what novel contributions can push the frontier?**

Our approach was grounded in cross-disciplinary ideas from dynamical systems, fluid mechanics, and information theory — prototyped cheaply on Apple Silicon, validated on A100, and refined on H100.

### What we built

**1. Low-Rank Q factorization (r=128) → 12 layers**

Inspired by PR #215's finding that Q matrices have extreme condition numbers (>100M), we factor Q as `dim→128→dim` per layer. This saves ~50% of Q params and makes each step ~8% faster. The speed savings fund a **12th transformer layer** — nobody else has gone to 12L yet.

The intuition from linear algebra: Q's effective rank is ~100-120 out of 512. The remaining singular values are noise that quantization destroys anyway. By factoring Q, we remove that noise at training time rather than losing it at quantization time.

**2. QAT with Straight-Through Estimator (int7)**

During training, we simulate int7 quantization in the forward pass using fake-quantize + STE for gradients. Activated at 10% of training (early — the model co-adapts with quantization noise from the start). 6% step time overhead.

The motivation: the quantization gap (pre-quant vs post-quant BPB) is one of the largest remaining sources of loss. QAT directly trains the model to be robust to it.

**3. FTLE-guided per-row precision (tested, negative result)**

We tracked per-row gradient sensitivity during training (a proxy for the Finite-Time Lyapunov Exponent from dynamical systems theory) and used it to allocate quantization precision per row — more bits for "hot" (sensitive) rows, fewer for "cold" rows.

**Result: uniform quantization is strictly better than FTLE-guided at every bit width.** Mixing int4 cold rows with int8 hot rows produces higher RMSE AND larger compressed size (mixed values have higher entropy → worse zstd compression). This is a clean negative result that saves future researchers from this path.

**4. Stride-OGD at eval (implemented, too slow)**

Online gradient descent on a 1024-dim vocabulary bias during sliding window evaluation. Zero artifact cost — the bias is computed from the eval text itself. The idea is sound (PR #241) but our implementation requires gradient tracking through [batch, 1024, 1024] logits tensors, which is prohibitively slow (~30-60 min for full eval).

## Research Pipeline

This submission is the output of a 3-stage research pipeline:

### Stage 1: Apple Silicon prototyping (18GB Mac)
- Created `make_mini_shards.py` for sub-1MB data subsets
- Tested layer sharing (depth recurrence): 3 shared blocks = 9 unique at 1/3 params
- Found optimal tiny config: 2 shared, 256d, MLP 3x, 1.45M params → 1.783 BPB locally
- Validated DEQ convergence theory: trained shared blocks become contractive (Lyapunov δ decreasing)
- Built FTLE sensitivity tracking infrastructure

### Stage 2: A100 validation (TACC Lonestar6, 1xA100 40GB)
- **Layer sharing abandoned at 512d** — costs 0.09 BPB vs unique layers (the 16MB budget already fits enough unique params)
- Integrated BigramHash + SmearGate → 0.094 BPB improvement
- Best A100 result: **1.3260 BPB** (9L, zstd-22, sliding window s1024)
- Identified 6 high-confidence improvements from competition PRs

### Stage 3: H100 refinement (1xH100 80GB)
- Implemented Low-Rank Q + 12 layers + QAT + FTLE + Stride-OGD
- Pre-quant val_bpb: **1.2035** at 7900 steps
- Clean negative result on FTLE per-row precision
- Stride-OGD needs optimization (too slow as-is)

## What We'd Do With a $500 RunPod Dev Grant

**Phase 1: 8xH100 validation ($25, ~2 hours)**
- Run our 12L + Low-Rank Q + QAT config on 8xH100 for proper scoring
- Expected: ~7800 steps in 10min at ~77ms/step
- A/B test: uniform int6 vs int7 (int7 = 16.9MB, need to trim 0.9MB via smaller code or BigramHash table)
- Target: sub-1.17 BPB post-quant with sliding window

**Phase 2: Hyperparameter sweep ($50, ~4 hours)**
- WD sweep: 0.03, 0.04, 0.05 (competition found 0.04 optimal)
- LR sweep: matrix_lr 0.020-0.035 (PR #198 uses 0.025)
- Muon momentum: 0.95, 0.97, 0.99 (PR #198 uses 0.99 with warmup from 0.92)
- SWA cadence: every 25, 50, 100 steps during warmdown (our continuous SWA hurt, periodic might help)
- Each A/B test: ~$3 per 10-min run

**Phase 3: Novel combinations ($100, ~8 hours)**
- 12L + int5-MLP/int6-attn mixed quantization (PR #180's technique + our 12L)
- QAT specifically for int5 MLP weights (nobody has combined QAT with int5/int6 mixed)
- Fix Stride-OGD eval speed (batch the gradient computation)
- Try 13 layers if Low-Rank Q speed savings allow
- Content-dependent pre-rotation (PR #215's promising-but-failed idea — we'd try a Triton kernel)

**Phase 4: Multi-seed validation + submission ($25)**
- 3-seed runs of the best config
- Statistical significance test (p < 0.01)
- Package records/ folder and submit PR

**Phase 5: Moonshot experiments ($300, remaining budget)**
- Stride-OGD + Two-Pass eval combined with full stack
- NTK-RoPE 4096 at eval (4x context without retraining)
- Adaptive Low-Rank Q (different rank per layer based on spectral analysis)
- BitNet b1.58 exploration (ternary weights for 5x more params in same space)

**Total: ~$200 for competitive submission, ~$500 to explore the frontier.**

## Files

- `train_gpt.py` — Full training script (1388 lines). Based on SOTA WarmdownQuantization record with Low-Rank Q, QAT, FTLE tracking, and Stride-OGD added.
- `EXPERIMENT_LOG.md` — Detailed H100 experiment log with training curves and ablations.
