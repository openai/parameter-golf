# V20: Cascaded 2-Phase L-BFGS Causal SLOT + Discriminative TTT

**3-seed mean: 1.00497477 BPB (1.69685330 nats)**

Beats merged SOTA PR #1019 (1.11473509 BPB) by 0.18532523 nats = **37.1x the required 0.005-nat threshold** (Welch t=-139.79, df=2.29, p<<0.001).

## The Stack

This submission layers one new eval-time optimization technique on top of the existing SOTA stack:

| Component | Source |
|-----------|--------|
| 11L backbone + SP1024 + XSA-all + BigramHash(3072,112) | PR #1019 (abaybektursun) |
| Full Hessian GPTQ int6 + brotli+lzma + Coprime loader + QK_GAIN=5.0 | PR #1019 |
| L-BFGS Causal SLOT eval loop (history reset per window, causal mask on already-scored tokens) | PR #1350 (resouer) |
| Discriminative per-block pre-quant TTT (graduated LR 0.3x→1.0x across 10 layer groups) | PR #1351 (resouer) |
| **Cascaded 2-Phase L-BFGS** (our addition) | This PR |

## What's New: Cascaded 2-Phase L-BFGS

The baseline L-BFGS Causal SLOT (PR #1350) runs a single 25-iteration L-BFGS pass per window with history_size=20. We split this budget into two phases:

- **Phase 1 (coarse):** 5 iters, history=10, uniform mean loss over the full 128-token focal window. Finds the dominant descent direction cheaply.
- **Phase 2 (refine):** 18 iters, history=20, uniform mean loss, **fresh L-BFGS instance with reset history**. Polishes the coarse Phase 1 solution.

**Why reset history between phases?** Though Phase 1 and Phase 2 share the same loss here (both uniform over the focal window), we designed the interface so Phase 2 can diverge (e.g., different weighting, focal window). Per Codex gpt-5.4 review: *"Previous L-BFGS curvature pairs approximate the old objective's Hessian. If Phase 2 changes the objective, those pairs are now approximating the wrong matrix. Warm-starting δ is good; inheriting the memory is usually not."* We warm-start the delta tensor across batches within an eval pass, but reset the L-BFGS memory between phases within a batch. We also warm-start δ across batches (proven useful from PR #1350).

**Why 5+18 = 23 iters (vs baseline 25)?** L-BFGS per-iter cost scales as O(history × dim). Total "history-iters":
- Baseline (single phase): 25 iters × history 20 = **500 history-iters**
- Cascaded V20: (5×10) + (18×20) = 50 + 360 = **410 history-iters**

This is ~18% less L-BFGS work with equivalent or better quality. In wallclock terms, SLOT eval drops from ~560s (PR #1350) to ~487s (V20) on 8xH100 — an 8% speedup on the eval phase.

**Implementation detail:** Phase 1 and Phase 2 both use uniform per-token loss over opt_mask positions. The opt_mask is strictly `[focal_start, s)` where `s = max(wl - slot_stride, 0)` — only already-scored positions from previous sliding windows. This is the same causality guarantee as PR #1350: test-time SLOT optimizes on tokens already graded, never on the positions currently being scored.

## Results

| Seed | val_loss (nats) | val_bpb | train_steps | artifact_bytes |
|------|-----------------|---------|-------------|----------------|
| 1337 | 1.69532641 | 1.00407045 | 6123 | 15,882,862 |
| 42 | 1.69939647 | 1.00648098 | 6122 | 15,832,250 |
| 999 | 1.69583703 | 1.00437287 | 6120 | 15,846,954 |
| **Mean** | **1.69685330** | **1.00497477** | 6121.67 | 15,854,022 |
| **Std** | 0.00221720 | 0.00131315 | — | — |

All 3 seeds trained to the 600s wallclock cap, landing at 6120-6123 training steps. Artifact sizes stay well under the 16MB (16,000,000 byte) cap across all seeds.

### vs PR #1019 (merged SOTA, 1.11473509 BPB ± 0.00035387)

| Metric | Value |
|--------|-------|
| Delta BPB | −0.10976032 |
| Delta nats | −0.18532523 |
| Welch t-statistic | −139.7872 |
| Welch df | 2.2890 |
| p-value | p << 0.001 |
| Threshold (0.005 nats) | 37.1x exceeded |

## Reproduction

```bash
# 1. Clone repo and download FineWeb (sp1024 variant)
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024

# 2. Install FA3 (Hopper FlashAttention 3) from source — required for 10-min budget
pip install ninja
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper && MAX_JOBS=8 python setup.py install
cd ../..

# 3. Copy this submission's train_gpt.py, then run 3 seeds
for SEED in 1337 42 999; do
    SEED=$SEED torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee run_$SEED.log
done
```

Expected: `final_causal_slot_exact val_bpb:` lines around 1.004-1.006.

## Hardware & Environment

- 8x NVIDIA H100 80GB HBM3 (SXM, RunPod secure cloud)
- PyTorch 2.4.1+cu124
- CUDA 12.4
- FlashAttention 3.0.0 (Hopper kernels, built from source)
- Total train + eval wallclock per seed: ~21 min (600s train + ~225s dTTT + ~486s SLOT)

## Ablation Notes

The V20 script also implements a second technique (importance-weighted CE mixture for Phase 2) gated behind `V20_GRAD_WEIGHTED=1` env var. That path uses `w_t ∝ (1 - p_target_t)` as the token-importance weight (Codex review: this is the true |dCE/dlogit| magnitude; NLL-weighting would over-concentrate on outliers). This submission runs with **`V20_GRAD_WEIGHTED=0` (uniform loss)** as the stable default. The importance-weighting path is intended for a future submission.

## Credits

- **PR #1019** (@abaybektursun) — backbone architecture, GPTQ, XSA
- **PR #1350** (@resouer) — L-BFGS Causal SLOT eval framework
- **PR #1351** (@resouer) — Discriminative per-block TTT
- Codex (gpt-5.4) — review of Cascaded L-BFGS history-reset rationale and importance-weighting correction
