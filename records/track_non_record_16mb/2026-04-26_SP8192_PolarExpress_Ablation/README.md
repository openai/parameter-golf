# Non-record: Polar Express NS Coefficient Ablation on SP8192 3-Layer Recurrence Stack

**Ablation study: Polar Express per-iteration Newton-Schulz coefficients vs fixed coefficients on PR #1809's architecture.**

**Result: Polar Express made things slightly worse (+0.00024 BPB).** The fixed NS coefficients `(3.4445, -4.775, 2.0315)` in #1809 outperform PE's per-iteration optimal coefficients for this architecture.

## Results (seed=42, 8×H100 SXM)

| Variant | val_bpb (TTT) | val_bpb (sliding) | val_bpb (quant, no TTT) | Pre-quant post-EMA | Steps | Train time | Artifact bytes |
|---------|---------------|-------------------|--------------------------|---------------------|-------|------------|----------------|
| **#1809 baseline** (fixed NS) | **1.08130** | 1.08262 | 1.09922 | 1.08805 | 4542 | 588s | 15,989,814 |
| **#1809 + PE5** (per-iter NS) | 1.08154 | 1.08303 | 1.09974 | 1.08825 | 4547 | 588s | 15,974,228 |
| **Δ (PE5 − baseline)** | **+0.00024** | +0.00041 | +0.00052 | +0.00020 | +5 | ~0s | −15,586 |

The PE variant is consistently worse across all evaluation modes (TTT, sliding window, quantized-only, pre-quant).

## Background

### What is Polar Express?

Polar Express (PE) replaces the fixed Newton-Schulz (NS) polynomial coefficients in the Muon optimizer with per-iteration optimal coefficients computed from the spectral radius of the current iterate. This was introduced by @orangekame3 in PR #1344 and applied in PR #1787 by @nprime06.

The fixed coefficients `(3.4445, -4.775, 2.0315)` are a single-point approximation to the optimal orthogonalization polynomial. PE computes the spectrally-optimal coefficients at each NS iteration, which should in theory give a better approximation to the orthogonal projection.

### Why test on #1809?

PR #1809 by @PranavViswanath (forking #1493 by @bigbag) holds a top leaderboard position (claimed val_bpb 1.08079) using an SP8192 tokenizer with a 3-layer depth-recurrence stack, INT5 mixed-precision QAT, zstd compression, and aggressive test-time training. It uses 5 NS steps in Muon. We tested whether PE could improve these NS steps.

### What we changed

In the PE5 variant, we replaced the fixed coefficients with per-iteration optimal coefficients computed via:
```python
spectral_radius = (X @ X.T).norm()  # Frobenius proxy
a, b, c = optimal_coeffs(spectral_radius)  # Per-iteration
```

Everything else (architecture, hyperparameters, data, seed, hardware) was held constant.

## Key Finding

**This is a negative result for PE at the same 5-step count.** The degradation is small (+0.00024 BPB with TTT) but consistent across all evaluation modes.

**Important limitation:** This ablation does *not* test the hypothesis that PE is useful because it enables fewer NS steps (e.g., 4 instead of 5), which saves optimizer wall time and yields more training steps within the 600s budget. As @PranavViswanath [noted](https://github.com/openai/parameter-golf/pull/1831#issuecomment-4322493813), the throughput gain from fewer steps — not better per-step orthogonalization — is the primary mechanism by which PE improves BPB in #1809.

Possible explanations for the same-step-count regression:

1. **The fixed coefficients are already well-tuned** for the spectral distribution encountered during #1809's training.
2. **PE's spectral radius estimation via Frobenius norm** may introduce noise that slightly degrades convergence.
3. **5 NS steps already provide sufficient orthogonalization** — the marginal improvement from optimal coefficients doesn't overcome the estimation overhead.

## Methodology

- Both runs used identical code (decoded from #1809's submission artifact) except for the NS coefficient computation.
- Same seed (42), same hardware (8×H100 80GB SXM), same data (FineWeb SP8192).
- Training was wallclock-capped at 600s; both runs completed ~4545 steps in ~588s.
- Single seed — this is an ablation study, not a SOTA claim. The delta (+0.00024) is below the statistical significance threshold (0.005 nats) required for SOTA claims.

## Attribution

- **@PranavViswanath** — PR #1809 (base architecture and training recipe, forking #1493)
- **@bigbag** — PR #1493 (upstream fork base)
- **@orangekame3** — Polar Express concept (PR #1344)
- **@nprime06** — PE integration in parameter-golf (PR #1787)

## Compliance

| Requirement | Status |
|-------------|--------|
| Artifact ≤ 16,000,000 bytes | ✅ 15,974,228 bytes (PE5) / 15,989,814 bytes (baseline) |
| Training ≤ 600s wallclock | ✅ ~588s both runs |
| 8×H100 SXM hardware | ✅ |
| No validation data during training | ✅ |
| Self-contained artifact | ✅ |

**Note:** This is a non-record submission. The baseline reproduction (1.08130) does not match #1809's claimed 1.08079, likely due to non-determinism across different H100 pod configurations. The ablation comparison is valid because both runs used the same pod setup and seed.

## Attempted Follow-Up (not executed)

Following @PranavViswanath's [methodology critique](https://github.com/openai/parameter-golf/pull/1831#issuecomment-4322493813), we prepared a partial follow-up ablation to test PE at the same 4-step count:

- **R1_fixed4_Gram** — fixed coefficients `(3.4445, -4.7750, 2.0315)`, 4 NS steps, Gram-NS on
- **R2_PE4_Gram** — Polar Express last-4 coefficients, 4 NS steps, Gram-NS on

This would isolate whether PE coefficients improve orthogonalization quality at the 4-step budget that #1809 actually uses. Note: this is a *partial* decomposition. The full suggested design (`fixed5 vs fixed4 vs PE4`) would also require a `fixed5` control to separate the step-count effect from the coefficient effect; `fixed5` was deferred due to time budget.

**Status:** Code was prepared and validated but the run was not executed. An 8×H100 SXM reservation was attempted on 2026-05-01 and failed with SUPPLY_CONSTRAINT on both SECURE and COMMUNITY clouds. No data was collected, no charges were incurred ($77.62 balance unchanged). The follow-up remains ready to execute when hardware becomes available.

**Conclusions remain limited to the PE5 vs fixed5 comparison above.**

## Files

- `train_gpt.py` — Decoded #1809 source (base variant with fixed NS coefficients)
- `train_seed42.log` — Full training log for PE5 ablation run
- `baseline_seed42.log` — Full training log for #1809 reproduction (fixed NS)
- `submission.json` — Submission metadata
- `README.md` — This file
