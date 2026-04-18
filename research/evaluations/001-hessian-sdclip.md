# Evaluation — Spec 001 (Hessian-SDClip λ screen)

**Run:** `runs/001-hessian-sdclip/` | **Hardware:** 1×H100 NA-1 | **Date:** 2026-04-19
**Code:** `exp/hessian-sdclip` @ `74c8385` | **Hotstart:** spec-000 `ckpt_final_pre_ema_step3849.pt`

## Result

6 λ values swept, quant-only eval, shared Hessian across all 6.

| λ | val_bpb_quantized | Δ vs λ=0 | artifact (bytes) | artifact MB |
|---|---|---|---|---|
| 0.00 | 1.10518 | — (baseline) | 15,979,642 | 15.24 |
| 0.05 | 1.10527 | +0.00009 | 15,980,308 | 15.24 |
| 0.10 | 1.10530 | +0.00012 | 15,982,510 | 15.24 |
| 0.20 | 1.10553 | +0.00035 | 15,991,263 | 15.25 |
| 0.40 | 1.10618 | +0.00100 | 16,019,404 | **15.28 ⚠ >16 MB** |
| 0.60 | 1.10676 | +0.00158 | 16,057,235 | **15.32 ⚠ >16 MB** |

**Signal gate (Δ ≤ −0.0003 for any non-zero λ): NOT MET.** All 5 non-zero λ values are *positive* Δ (worse than control).

## Noise/signal judgment — clear kill signal

Not noise. Monotonic worsening with increasing λ across a 12× range (0.05 → 0.60), at 4+ distinct data points. That's a strong, consistent signal that the technique actively hurts on our stack.

Effect magnitudes (smallest +0.00009, largest +0.00158) may look small relative to SOTA's ~0.0002 std, but: (a) the trend is monotonic not random, (b) the worst case is ~8× SOTA std, (c) the Δ is reproducible across λ values, not a one-seed fluke.

This is not a "tune it smaller" situation — going from λ=0.60 → 0.40 → 0.20 → 0.10 → 0.05 just takes us back toward λ=0 (no-op). There's no sweet spot below our tested range that could flip the sign.

## Secondary finding — artifact size coupling

At λ=0.40 and 0.60, the compressed `.ptz` artifact exceeds the 16,000,000-byte leaderboard limit. **So even if a wild high-λ had shown bpb improvement, it would have been inadmissible as a submission.**

Mechanism: the `adj = 1 + λ(r_i − 1)` multiplier widens the distribution of per-row clip scales `s`. Wider scale distribution → lower Brotli compression efficiency on the int6 matrices → larger artifact. This is a *general* property of per-row scale modulation, not specific to Hessian-SDClip.

Implication beyond this spec: any future quant-time candidate that stretches per-row or per-group scales (e.g. per-group bit allocation at higher precisions) needs to track both bpb AND artifact size, with explicit budget gating.

## Methodology finding — 1×H100 ≠ 8×H100 absolute bpb

Spec 001's validity gate (λ=0 reproduces spec-000's 1.10430 within ±0.0001) **did not hold**: we landed at 1.10518, off by +0.00088. After investigation, this is NOT a code bug but an expected consequence of `ShuffledSequenceLoader` distributing calibration data across DDP ranks:

- 8×H100 (spec 000): 64 calibration batches spread across 8 ranks, each seeing a different FineWeb shard.
- 1×H100 (spec 001): all calibration data from rank-0's shard only.
- Different calibration data → different Hessian diagonal → different GPTQ error correction (`Hinv` / Cholesky) → different quantized weights → different val_bpb, *even on the λ=0 no-op clip path*.

**Intra-sweep Δ (λ=0.05/.../0.60 vs our λ=0) is still valid** — same Hessian used across all 6 configs. All reported Δs are sound.

**Cross-hardware absolute bpb is NOT valid.** Spec 002 onwards will use in-sweep control as baseline, not any spec 000 number.

## Loss-curve notes
N/A — no training in this screen.

## Cost accounting

- Spec estimate: ~$0.45 on 1×H100.
- Actual: **~$1.90** (~4× over).
- Overrun cause: a device-mismatch bug in the sweep.py wrapper (`map_location=device` when re-loading cached Hessian; should be `"cpu"`) crashed round 2 mid-sweep, forced a fresh compile on re-entry. One-character fix. Correct pattern ported to spec 002's `swa_sweep.py`.
- Lesson: a 30-second local dry-read of exp-branch scripts would catch this class of bug before shipping. Building that into pre-push habit.

## Decision: **KILL**

- Idea `research/ideas/hessian-sdclip.md` updated with `Status: ❌ SHELVED 2026-04-19`.
- Do not revisit without a fundamentally different formulation (e.g. per-column adjustment instead of per-row, or a non-linear `adj` function that doesn't widen the scale distribution).
- Don't iterate with different λ ranges, different row-importance metrics, or different calibration data — the monotonic-worsening pattern is definitive.

## Next steps

1. **Spec 002 (SWA + EMA) proceeds as already frozen.** Different mechanism (flatter minima via uniform averaging) — fails for different reasons if it fails. Also doesn't suffer from the per-row-scale-widening problem. Baseline in spec 002 is already calibrated to the 1×H100 Hessian regime (~1.10518).

2. **Post-training candidate ceiling is likely lower than original plan estimates.** SOTA's quant pipeline is more fully tuned than I'd assumed. If SWA+EMA also fizzles, the next move is probably to a training-time candidate (BigramHash first, since the code already exists).

3. **Artifacts retained on NA-1 volume:** `hessians.pt` (232 MB, reusable for any future Hessian-based experiment on `ckpt_final_pre_ema_step3849`). `.ptz` files (6 × ~16 MB) — probably useless given the negative result but kept through the record-track push in case the 1.10518 baseline artifact is useful as a reference.

## Running tally
- Spent: $14.90 of $200 hard budget.
- Remaining: ~$185.
- Days to deadline: 11 (through 2026-04-30).
- Candidates: 1 killed, 1 in-flight (spec 002), 3 behind.
