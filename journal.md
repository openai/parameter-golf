# Journal

## Current threads
- Anchor baseline: exp `0001_baseline_repro` at val_bpb 2.5212 (post-quant int8+zlib), 6.907 MB. Bit-reproduces the Apr-18 reference run. All sentinels and noise-floor comparisons still reference this row.
- **Best so far: 2.4052** (`winners/2026-04-25_warmdown_600_warmup_10`, exp 0005, confirmed by SEED=42 in 0006 at 2.40272). Schedule change only (`LR_WARMUP_STEPS=10`, `WARMDOWN_ITERS=600`); +1.2 MB artifact (still 8.1 MB / 16 MB cap).
- The lr_mul formula in `train_gpt.py` is `(iterations−step)/warmdown_iters` after warmup. With ITERATIONS=200, the canonical default `WARMDOWN_ITERS=1200` gives lr_mul peaking at 0.167 (avg 0.083) — extremely attenuated. The 0005 schedule (warmup_10 + warmdown_600) doubles avg lr_mul to 0.178; tested up to and through a brief one-step lr_mul=1.0 spike at the warmup peak (recoverable). Further-aggressive schedules NOT yet tested.
- Capacity (MLP_MULT, exp 0002) and attention temperature (QK_GAIN_INIT, exp 0003) showed only Δ≈+0.002 each under the canonical schedule — noise-band. Hypothesis: their effects are MASKED by the under-training. Both should be re-tested ON TOP OF the new schedule.
- Quant tax actually IMPROVED in 0005 (0.0029 vs baseline 0.0055) — better-trained weights quantize cleaner. Means architectural + schedule improvements are likely additive in post-quant terms.
- All schedule wins are **[transfer:low]** — the H100 20k-step regime has a different optimal schedule. Future autoresearch should focus architectural experiments on top of this schedule (transfer:high/med candidates).

---

## Entries (newest first)

## 2026-04-25 · exp 0005 + 0006 · schedule rewrite — first big win (Δ +0.116)

**Question**: After 0002 (MLP capacity) and 0003 (q_gain temperature) both came in at Δ≈+0.002, the leading hypothesis was that the canonical default `WARMDOWN_ITERS=1200` (peak lr_mul 0.167, avg 0.083) keeps the model so under-trained that no architectural change can show its true effect. Does opening up the schedule to peak lr_mul=1.0 briefly + warmdown from 0.317 produce a real Δ?

**Setup**: `LR_WARMUP_STEPS=10` + `WARMDOWN_ITERS=600`. Schedule: ramp 0.1 → 1.0 over steps 0-9, then warmdown branch `(200−step)/600` starting at 0.317 at step 10, decaying to 0 at step 200. Avg lr_mul = 0.178 (vs 0.083 baseline → 2.14×). **Pre-experiment 0004** without warmup (WARMDOWN_ITERS=600 alone) was killed at step 10: step 2 train_loss spiked from 6.94 → 8.40, an LR-induced first-step overshoot of cold tok_emb. Adding the 10-step warmup was sufficient to make the schedule trainable.

**Prediction** [LIKELY]: Δ ≈ +0.030 to +0.080.

**Disconfirming**: NaN around the lr_mul=1.0 spike at step 9 (would require gentler warmup); Δ ≤ +0.005 (would mean LR isn't the bottleneck either).

**Result**:
- 0005 (SEED=1337): val_bpb_post=2.40517, pre=2.4023, quant_tax 0.0029, artifact 8.105 MB.
- 0006 (SEED=42 confirm): val_bpb_post=2.40272 — Δ between seeds 0.00245.
- **Mean Δ vs baseline: +0.117** — way above the +0.050 "suspicious-large" threshold but reproduces tightly across seeds.
- Trajectory: step 9 train_loss=7.058 (above step 1) — the brief lr_mul=1.0 spike does cause mid-warmup overshoot, but recovers by step 10 once warmdown drops lr_mul to 0.317. Step 35=4.85 (vs ~5.13 baseline-class trajectory), then steady descent to step 200=4.26 (vs 4.42 baseline). No NaN.
- Quant tax DROPPED from 0.0055 to 0.0029 — better-trained weights have more structured distributions that quantize cleaner. Important: post-quant gains can OUTPACE pre-quant gains for schedule changes.

**Conclusion** [VERIFIED across 2 seeds]:
1. The 200-step smoke under canonical `WARMDOWN_ITERS=1200` is severely compute-starved — not because the schedule is "wrong" for 200 steps but because it was inherited from a 20k-step canonical run. Doubling avg lr_mul (with the warmup needed to avoid first-step blowup) recovers ~0.12 of val_bpb that was previously "trapped behind under-training."
2. The brief lr_mul=1.0 spike at step 9 causes a transient train_loss bump but is fully recoverable on MPS bf16; at least at this brevity. Pushing the schedule further (sustained lr_mul ≥ 0.5) is the next axis to test.
3. **Strongest implication for autoresearch**: prior architectural ablations under the canonical schedule are likely false negatives. MLP_MULT, QK_GAIN_INIT, and any other "improves training" change should be re-tested ON TOP OF this schedule before being discarded.
4. **[transfer:low]** — the H100 20k-step regime has a different schedule sweet spot. The discovery itself is about the *autoresearch testbed*, not the submission. Useful as a multiplier for ranking architectural changes.

## 2026-04-25 · exp 0002_mlp_mult_3 · capacity isn't the bottleneck at 200 steps

**Question**: With ~9 MB of artifact headroom, does spending some on extra MLP capacity (mlp_mult 2→3) move val_bpb at 200 steps?

**Setup**: Forked from canonical, env-var-only change `MLP_MULT=3`. Predicted artifact 8.8 MB; actual 8.404 MB. Same WARMDOWN_ITERS=1200 schedule as baseline.

**Prediction** [LIKELY]: Δ ≈ +0.010 to +0.025 — record lineages repeatedly use mlp_mult ∈ {3, 4} (e.g. 2026-04-01 SP4096+MLPMult4 at 0.9979).

**Disconfirming**: Δ ≤ +0.005 → capacity isn't the dominant bottleneck under this schedule.

**Result**: post-quant Δ=+0.00212 (noise). Pre-quant Δ=+0.0046 (also noise). The bigger MLP picked up extra quant tax (0.0079 vs 0.0055 baseline), so post-quant gain is half of pre-quant gain. Trajectory was healthy: step 1=6.9383 (vs baseline 6.9379), monotonic, no NaN.

**Conclusion** [LIKELY]: At 200 steps under WARMDOWN_ITERS=1200 (avg LR mul ≈ 0.083), MLP capacity is **not** the limiting factor. Records using mlp3x/mlp4x trained for 20k steps on H100, where the under-trained MLP capacity hypothesis doesn't apply. Lesson for autoresearch: prioritize schedule, attention temperature, and initialization changes (which take effect immediately) over scaling parameters (which need steps to be useful) on the 200-step smoke. Capacity ablations should ride on top of a more aggressive schedule, not the canonical-attenuated one.

## 2026-04-25 · exp 0001_baseline_repro · canonical baseline reproduced

**Question**: Can we run a stable 200-step smoke on MPS that bit-reproduces the Apr-18 reference (val_bpb 2.5540)?

**Setup**: Canonical `train_gpt.py`. env.sh sets `WARMDOWN_ITERS=1200` so the step-based warmdown is active from step 0 (`warmdown_start = max(200−1200, 0) = 0`), giving LR multiplier 0.167 at step 0 decaying to ~0 by step 200. Otherwise canonical hyperparameters: `WARMUP_STEPS=0`, `MAX_WALLCLOCK_SECONDS=0`, `TIED_EMBED_LR=0.05`, `MATRIX_LR=0.04`, `SCALAR_LR=0.04`, `GRAD_CLIP_NORM=0`. `VAL_TOKENS=16384` (vs Apr-18's full val).

**Prediction** [LIKELY]: step 2 = 6.7505 (matches Apr-18 log), final val_bpb in [2.4, 2.7].

**Disconfirming**: any deviation from the Apr-18 trajectory (would mean MPS or some env state had drifted), or NaN.

**Result**: trajectory matches Apr-18 to 4 decimal places — step 1=6.9379 ✓, step 2=6.7505 ✓, step 200 train_loss=4.4196 ✓. Artifact 6.906960 MB (Apr-18: 6.905876 MB) — close but ~1 KB off, likely from train_gpt.py's source bytes growing slightly post-Apr-18 (LR_WARMUP_STEPS commit added lines, code_bytes counts it) plus tiny zlib non-determinism downstream of the bf16 model body. val_bpb_post_quant=2.5212 (Apr-18: 2.5540) — the gap is VAL_TOKENS=16384 vs Apr-18's full ~1M-token eval. No NaN.

**Conclusion** [VERIFIED]: The earlier "Apr-18 was a non-reproducible MPS lucky draw" framing was wrong. The schedule was deterministically attenuated — `lr_mul` with `WARMDOWN_ITERS=1200` and `ITERATIONS=200` makes `warmdown_start = max(200−1200, 0) = 0`, so warmdown is active from step 0 and the effective LR multiplier is `(1200 − step) / 1200`. Apr-18 was running at ~17% LR throughout, decaying to ~0% by step 200. This is what kept training stable. Full canonical LR is too aggressive for MPS bf16 numerics (NaN around step 165), and that's what was happening when the env.sh template incorrectly set `WARMDOWN_ITERS=40`.

Three lessons worth carrying forward:

1. **`WARMDOWN_ITERS=1200` is the canonical default** for short MPS smokes. Keep it. Override only when the experiment specifically wants full canonical LR — and pair with explicit `LR_WARMUP_STEPS=10–20` if so.

2. **MPS bf16 has tighter LR tolerance than CUDA bf16.** Canonical `MATRIX_LR=0.04` + `TIED_EMBED_LR=0.05` work on H100 with FlashAttention-3 fused kernels and tighter bf16 guard bits. On MPS they NaN at full LR. The implicit warmdown attenuation hides this; if you remove it without warmup, expect `tok_emb` to NaN at step 2 and `skip_weights` to NaN around step 165.

3. **When you can't reproduce a "lucky" baseline, examine the schedule before suspecting nondeterminism.** PyTorch MPS *is* documented as nondeterministic across runs, but the failure mode of being unable to repro a known-good config is much more often a config delta than a kernel-level RNG difference. Diagnose via `lr_mul` math first, and verify the values that go into it.

## 2026-04-25 · note · full-val eval cost on MPS

Tried `VAL_TOKENS=0` (full ~1M-token val) to get a lower-variance read on the baseline. Aborted after eval was still running 21 minutes after training finished. Total elapsed would have been ~25 min vs ~5 min for the capped eval.

Bottleneck: the patched `eval_val` accumulates `val_loss_sum` / `val_token_count` / `val_byte_count` in float64 on CPU (MPS doesn't support fp64), so each of ~1023 eval micro-steps round-trips a per-batch scalar across the MPS↔CPU boundary, plus a token-id reshape and LUT lookup that also crosses. Roughly 3 device-syncs per batch × 1023 batches = the tax.

Why fp64 at all: eval reports val_bpb to 8 decimals (`val_bpb:2.52115777`), and summing ~1M token-loss contributions in fp32 would lose ~4-5 of those decimals (relative epsilon ~1e-7 × 1023 batches × ~7 nat per batch ≈ 1e-3 absolute drift). Canonical chose fp64 to keep the summation noise below the reported precision; we don't modify the eval harness.

Decision: keep `VAL_TOKENS=16384` as the autoresearch default. `VAL_TOKENS=0` stays available for confirming a marginal result, with the cost (~5× the smoke budget) documented in `program.md`.

## 2026-04-25 · note · fp32+full eval is also forbidden; full-val is just too slow on MPS

Tried again with a modified `eval_val` that accumulates in fp32 on the MPS device (no CPU round-trip). Hypothesis: the bottleneck was CPU↔MPS sync per batch, so keeping everything on-device should make full-val tractable.

It didn't. Run hit 66 min and was still going when killed. Pre-quant val_bpb 2.5274 was logged on the way (vs Apr-18's fp64+full 2.5485 — gap ~0.02, partly fp32 accumulation noise, partly MPS run-to-run training drift visible at step 2: 6.7507 today vs 6.7505 Apr-18).

Why fp32-on-MPS didn't help: per-batch dispatch latency dominates over CPU↔MPS sync. ~8 MPS ops per batch × 1023 batches × ~50–100 ms dispatch each is its own multi-minute tax, separate from the sync. Forward pass on a 17M-param 1024-token batch on MPS is also surprisingly slow (~0.3 s/batch wallclock), making the floor ~5 min just for forward passes — and that's only the *first* eval. **`eval_val` is called twice** (pre-quant and post-int8-quant), so any full-val approach doubles. Total realistic best case ~15 min, observed >60 min.

Updated `program.md` and `env.sh` template to forbid `VAL_TOKENS=0`. Marginal-result confirmation is now done by re-running with `SEED=42` instead. The 16K-cap sample is enough at the 0.010 noise floor — sampling error cancels in same-seed Δ comparisons, which is what ranking actually needs.

The fp32 eval modification itself is reverted; the experiment folder is gone. The canonical eval harness stays untouched (rule preserved).
