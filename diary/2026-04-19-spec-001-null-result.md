# 2026-04-19 — Spec 001 null result + shelving Hessian-SDClip

Spec 001 (Hessian-SDClip λ sweep) ran on 1×H100 NA-1. 6 λ values: {0.00, 0.05, 0.10, 0.20, 0.40, 0.60}. Result below.

## What happened

Screened all 6 configs with shared Hessian, no training, quant-only eval.

| λ | val_bpb_quantized | Δ vs λ=0 | artifact |
|---|---|---|---|
| 0.00 | 1.10518 | — | 15.24 MB |
| 0.05 | 1.10527 | +0.00009 | 15.24 MB |
| 0.10 | 1.10530 | +0.00012 | 15.24 MB |
| 0.20 | 1.10553 | +0.00035 | 15.25 MB |
| 0.40 | 1.10618 | +0.00100 | **16.02 MB ⚠** |
| 0.60 | 1.10676 | +0.00158 | **16.06 MB ⚠** |

**Monotonic worsening.** Not just "no signal" — actively hurts, more λ is worse. Not ambiguous, not borderline. Kill signal.

## What the null actually means

Three plausible reasons Hessian-SDClip didn't transfer from the near-SOTA (2026-04-06) submission to our stack:

1. **SOTA's SDClip is already near-optimal** for current architecture. The row-importance signal is redundant — not adding information, just adding noise.
2. **The near-SOTA win was on a weaker base.** That submission's starting bpb was 1.0835, more headroom for small quant improvements. At our 1.086 post-SDClip-on-current-SOTA, we may already be at the post-training ceiling.
3. **The `adj = 1 + λ(r − 1)` formulation introduces per-row scale variance** that interacts poorly with GPTQ's column-wise error correction at our model scale.

We can't distinguish between these without more work, and it doesn't matter — the data says "stop." Shelving.

## Two side-findings that matter more than the null

### Artifact size grows with λ
At λ=0.40 and 0.60, the `.ptz` artifact exceeds the 16MB leaderboard limit. Cause: the row-wise scale multiplier `adj` broadens the distribution of per-row scales, which reduces Brotli's ability to compress the quantized int6 matrices.

**This is a general lesson, not just a Hessian-SDClip quirk:** bpb and artifact size aren't independent in quant-time modulations. Anything that stretches per-row clip scale likely hurts Brotli. So even if one of those high-λ runs had *improved* bpb, it'd be disqualified. The technique's upside is capped by the budget at the same time the downside (bpb) is accumulating.

Worth remembering for per-group bit allocation if we ever revisit it — that's another quant-time modulation that could hit the same coupling.

### 1×H100 Hessian ≠ 8×H100 Hessian
Our λ=0 landed at 1.10518, not spec-000's 1.10430. A ~+0.0009 offset without any code change. Root cause: `ShuffledSequenceLoader` distributes calibration data across DDP ranks, so an 8-GPU Hessian sees 8 different FineWeb shards (1 per rank), while a 1-GPU Hessian sees only the rank-0 slice. Different calibration data → different Hessian diagonal → different GPTQ error correction (`Hinv` Cholesky) → different quantized weights → different val_bpb, *even on the λ=0 no-op clip path*.

**Methodology implication:** intra-screen Δ is valid; cross-hardware absolute bpb is not. For every future 1×H100 screen, treat the in-sweep control as the baseline, not any 8×H100 number.

I've already baked this into spec 002 — its validity gate expects ~1.10518, loose ±0.0005, not ~1.10430 tight.

## Cost note

Spec estimated ~$0.45. Execution paid **$1.90** — ~4× over. Cause: a device-mismatch bug in the sweep.py wrapper (`map_location=device` vs `"cpu"` for cached Hessians) crashed round 2 mid-sweep, forcing a fresh compile on re-entry. One-character fix, but it cost ~$1.50 in pod time.

Lesson for planning: even trivial sweep wrappers have foot-guns. A 30-second local dry-read before shipping would have caught this. I've ported the correct pattern into spec 002's `swa_sweep.py` so we don't eat the same bug twice.

## Reusable infrastructure the run produced

Two pieces worth keeping around:
- `hessians.pt` on NA-1 volume — 232MB, 64 calibration batches worth of Hessian diagonals on the spec-000 `ckpt_final_pre_ema_step3849` weights. Any future Hessian-based experiment on the same checkpoint can skip collection.
- `sweep.py` pattern — idempotent single-checkpoint sweep harness. Spec 002's `swa_sweep.py` is a direct derivative. Any future quant-time / eval-time screen on a fixed checkpoint should reuse this shape.

## Effect on strategy

- **Hessian-SDClip shelved.** Not promoted, not iterated, not revisited. Monotonic-worsening is definitive.
- **Post-training ceiling is lower than I estimated.** If SWA+EMA also fizzles, we're essentially forced to BigramHash or one of the training-time candidates for the record.
- **Spec 002 (SWA+EMA) stays on path.** Different mechanism — generalization via flatter minima, not per-row clip modulation. It would fail for different reasons if it fails. Also: SWA-style averaging doesn't stretch per-row scales, so the bpb/artifact-size coupling problem shouldn't apply.

Updated 4-candidate queue → 3:
- ~~Hessian-SDClip~~ (killed)
- SWA + EMA (spec 002, ready for execution)
- BigramHash (full retrain; largest claimed Δ but expensive and budget-risky)
- Progressive recurrence (design fork unresolved)
- Per-group bit allocation (considering — the artifact-size-coupling lesson above makes me more skeptical)

## Running tally

- Spent: $13 (spec 000) + $1.90 (spec 001) = **$14.90 of $200**.
- Days: 11 used (through today), 11 remaining to 2026-04-30.
- 1 candidate killed, 1 in-flight, 2-3 viable behind it.
- Still zero improvements over our 1.08622 baseline — we're in the middle of the "spending budget to rule things out" phase.

## Takeaways to bank

1. **Hot-start screens are working as designed.** $1.90 to rule out a candidate with a clear kill signal is cheap. The method is sound even when a specific candidate isn't.
2. **Pre-screening absolute-bpb expectations against hardware configuration matters.** Spec 001's validity gate was mis-calibrated because I didn't think about GPTQ's per-rank calibration data. Fixed in spec 002; worth remembering for all future post-training screens.
3. **Coupling between bpb and artifact size is real.** Future quant-time candidates need to report both, with explicit budget gating.
4. **Build the dry-read habit before shipping exp code.** The device-mismatch bug would have been caught in 30 seconds of reading. I'll add a "dry-read self-check" step before pushing exp branches.
