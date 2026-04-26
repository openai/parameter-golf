# Experiment 0013_batch_16k_on_winner

Parent: canonical (batch-size control for 0012)

## Question
0012 doubled both `TRAIN_BATCH_TOKENS` (8192→16384) and `TRAIN_SEQ_LEN`
(1024→2048). Δ vs 0008 = +0.0228. To attribute the gain, run with
batch=16384 but seq_len=1024 (same as 0008 except batch).

If 0013's Δ vs 0008 ≈ 0.0228 → batch was the lever, seq_len contribution is small.
If 0013's Δ vs 0008 ≈ 0 → seq_len was the lever; 0012's gain was real "longer context."
If 0013's Δ ∈ [0.005, 0.015] → both contribute, roughly evenly.

The math: under a fixed lr_mul schedule, doubling batch tokens means each
optimizer step sees 2× more gradient signal — same direction as more steps,
without the schedule attenuation cost. This often dominates early-training
gains.

## Hypothesis [LIKELY]
Δ vs 0008 ≈ +0.012 to +0.020 — batch_tokens carries most of 0012's gain.
Bigger batches in early training are a near-universal lever; seq_len gain
should be smaller because the model is small (d=512) and 1024 vs 2048
context is mostly captured by the first 1024 tokens for most useful patterns.

## Change
`env.sh`:
- `LR_WARMUP_STEPS=10`
- `WARMDOWN_ITERS=600`
- `MLP_MULT=4`
- `TRAIN_BATCH_TOKENS=16384`
(seq_len stays at default 1024; VAL_BATCH_SIZE stays at default 8192 since
seq=1024 fits the 1024-token-per-microstep slot.)

## Disconfirming
- Δ ≈ 0.0228 (matches 0012): batch carried it all. Then 0012's promotion
  is properly framed as "bigger batch + nominal cost" rather than "longer
  context."
- Δ ≈ 0 (no gain from batch): 0012's Δ was entirely from seq_len=2048.
  Then we should still consider running pure-seq experiments (which
  requires a code change to grad_accum_steps).
- NaN: bigger batch with current LR is unstable. Would suggest needing
  longer warmup or smaller per-optimizer LR.

## Notes from execution
- val_bpb_post=2.30956 vs 0008 winner 2.39135 → **Δ=+0.0818**.
- val_bpb_post=2.30956 vs 0012 winner 2.36857 → Δ=+0.0590 (0013 strictly dominates 0012).
- Pre-quant Δ vs 0008: +0.0805 (2.3864 → 2.3059).
- quant_tax 0.0037 (in line; no quantization regression from bigger batch).
- artifact 11.753 MB (same as 0008's 11.774; no new params).
- step_avg 2578 ms — about 1.9× slower than 0008's 1360 ms (close to the
  2× expected from doubled batch). Notably *faster* than 0012's 3432 ms
  because seq=1024 avoids the attention² cost.
- Trajectory: step 1=6.9379 (canonical), warmup spike at step 9 similar to
  0008, then descent. Final step 200 train_loss substantially below 0008's.

**Decomposition of 0012's gain**:
- 0012 (batch=16k, seq=2048): 2.36857 vs 0008 baseline 2.39135 → +0.0228.
- 0013 (batch=16k, seq=1024): 2.30956 vs 0008 baseline 2.39135 → +0.0818.
- Implied seq=1024 → seq=2048 effect at batch=16k: −0.0590 (HURTS).

**Status: parked**. Δ vs 0008 (+0.082) and Δ vs 0012 (+0.059) both exceed
the +0.050 "suspiciously large" threshold per program.md. Must SEED=42
confirm before promoting. 0014 will be the SEED=42 sibling.

**Promotion implication**: 0012's promotion is technically correct (it
beat 0008), but 0013 strictly dominates 0012 with the longer-seq overhead
removed. Once 0014 confirms, 0013 should be the canonical winner; 0012
stays in `winners/` as historical record.
