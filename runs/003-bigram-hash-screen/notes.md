# Execution notes — spec 003

## Outcome
Training completed naturally (wallclock cap at step 4544, 2388s train_time). Pre-quant post-EMA val_bpb = **1.08788** landed. Signal gate failed by +0.00318 vs target ≤ 1.0847. GPTQ crashed immediately after pre-quant eval (see "Bugs caught" below) — but that was post-signal-gate, no impact on the research question.

## Timeline (UTC, pod `xo0yn9wbyjhe4u`)
- `23:27` pod `xo0yn9wbyjhe4u` created (2×H100 NA-1, $5.98/hr)
- `23:27-28` preflight: brotli install, git checkout `3825019`, data + tokenizer verified, 2×H100 confirmed
- `23:28` training launched via setsid
  - First 5 steps: variant train_loss nearly identical to Exp 24 (step 5: ours 8.3124 vs exp24 8.3028 — zero-init BigramHash is effectively no-op)
  - Step 100: variant WORSE by +0.0373 (brief BigramHash adaptation phase)
  - Step 300: variant catches up (Δ +0.0025)
  - Step 500: variant slightly BETTER (Δ −0.0052)
  - Step 2500-4500: variant consistently WORSE (Δ +0.003 to +0.006 across warmdown)
- `~00:07` variant hits step 4500 at ~39.4m train_time
- `00:09:30` stopping_early wallclock_cap @ step 4544
- `00:09:30` EMA apply + pre-quant eval → **val_bpb 1.08787912** written to log ✓
- `00:09:44` GPTQ crash: `KeyError: 'bigram.embed.weight'` (bug — see below)
- `00:15` rsync + `pod stop` + `pod delete`

## Cost
- Pod start 23:27 → delete 00:15 = **~48 min at $5.98/hr ≈ $4.80 pod rental**
- Actual balance drop: $25.37 → $20.26 = **$5.11** (pod + small volume-storage charge)
- Spec estimated $4.00; actual $5.11 (~28% over). Overhead mostly from our pod being ~7-10% slower in compile/early-training phase (caught up in steady state).

## Matched-step comparison methodology worked

Opened the interview with user about step-matched vs wallclock-matched comparison. User's instinct was to go wallclock-matched (penalize BigramHash's per-step cost directly). I started by agreeing, then retracted and endorsed the spec's original step-matched recommendation because:

1. Exp 24 ran on a different (faster) pod. A wallclock-matched comparison would conflate BigramHash's intrinsic cost with our pod's variance.
2. BigramHash's per-step compute overhead is theoretically tiny (~0.003% of forward pass — one 3072×112 embedding lookup + one 112→512 projection per token).
3. Our pod ended up catching up by end of training — final step count was 4544 vs Exp 24's 4531, so step-matched AND wallclock-matched comparisons coincide at the final eval anyway.

In practice it didn't matter — variant reached ~same final step as Exp 24 despite early compile-overhead drag. Both comparison philosophies converge on the same kill verdict here.

## Bugs caught / lessons

### 1. `gptq_mixed_quantize` crashes when `BIGRAM_VOCAB_SIZE>0`

After the pre-quant post-EMA eval landed (and wrote the signal number we needed), the script continued into GPTQ:

```
GPTQ:collecting Hessians from calibration data...
GPTQ:collected 67 Hessians in 13.0s
KeyError: 'bigram.embed.weight'
```

**Root cause:** `gptq_mixed_quantize` (`train_gpt_sota.py:848-862`) treats any state_dict entry > 65536 params as a quantization candidate. `bigram.embed.weight` is 3072 × 112 = 344,064 params → qualifies. But `collect_hessians` only hooks `CastedLinear` modules, not `nn.Embedding`. So `hessians['bigram.embed.weight']` doesn't exist → KeyError.

**Fix** (for any future BigramHash revival — research's call):
- Option A (simpler): special-case `bigram.embed.weight` as `passthrough (float16)` — one-line name check alongside the existing `tok_emb` branch.
- Option B: add an `nn.Embedding` hook in `collect_hessians` so bigram gets its own Hessian. More code, but keeps bigram quantized to int6.

The SOTA submission always used `BIGRAM_VOCAB_SIZE=0`, so this code path was never exercised. Not a bug caught by testing — caught by our first non-zero sweep.

### 2. Forgot to set `VAL_LOSS_EVERY=200` to match Exp 24

Our variant ran with the code default `VAL_LOSS_EVERY=4000`, meaning only one mid-training val was sampled (step 4000) plus the final step val. Exp 24 ran with `VAL_LOSS_EVERY=200`, giving a much richer val_bpb trajectory.

Impact: **none on the signal gate** (only end-of-training pre-quant val is required). But we lost the nice-to-have mid-training `Δ_val` column for the comparison table.

**Takeaway for future specs:** if a spec references an existing log as its baseline, execution should grep the baseline log for which env vars it was run with (esp. `VAL_LOSS_EVERY`, `TRAIN_LOG_EVERY`) and mirror them even if the spec doesn't explicitly list them.

### 3. Early-training throughput drag amortizes

Variant was 8-10% slower per step at steps 100-300 (~1.67 vs 1.85 steps/sec). By step 4200, rate had caught up to Exp 24's ~1.92 steps/sec. Compile warmup dominates the first few minutes on a fresh container disk.

**Implication for monitor:** don't alarm on early tok/s differences. Wait until ~step 1000 before projecting final step count.

## Handback

- `summary.md` — full results table, signal-gate analysis, Δ breakdown by training region.
- `variant_train.log` — full training log + GPTQ crash traceback.
- `notes.md` — this file.

Research: evaluation writeup + `experiments.md` row + kill decision is yours. My read matches the signal-gate math: **kill, shelve BigramHash for the April stack** (spec 003's hypothesis is falsified). If ever re-attempting, note the `bigram.embed.weight` GPTQ bug needs a one-line fix first.
