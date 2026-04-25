# Journal

## Current threads
- Anchor baseline: exp `0001_baseline_repro` at val_bpb 2.5212 (post-quant int8+zlib), 6.907 MB. Bit-reproduces the Apr-18 reference run. All sentinels and noise-floor comparisons reference this row.
- Best so far: 2.5212 (the baseline). Promote any experiment whose val_bpb is meaningfully lower (Δ ≥ +0.010 — see noise floor).
- MPS bf16 numerics can't tolerate the full canonical LR; the env.sh default `WARMDOWN_ITERS=1200` (≥ ITERATIONS=200) gives an effective LR ramp of (1200−step)/1200 across the run, which is what makes training stable. Don't drop this default without setting an explicit `LR_WARMUP_STEPS`.

---

## Entries (newest first)

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
