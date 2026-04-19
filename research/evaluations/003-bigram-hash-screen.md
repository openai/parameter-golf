# Evaluation — Spec 003 (BigramHash signal screen)

**Run:** `runs/003-bigram-hash-screen/` | **Hardware:** 2×H100 NA-1 | **Date:** 2026-04-19
**Code:** `research` @ `3825019` (no code change; hyperparam-only) | **Baseline:** `logs/exp24_sota_sp8192_2xh100_40m.log` (matched-step comparison, no re-run of control)

## Result

Single 2×H100 training run at spec-000 config **except** `BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112`, with `QK_GAIN_INIT=5.0 TTT_ENABLED=0 SEED=1337 MAX_WALLCLOCK_SECONDS=2400` to match Exp 24 exactly. Ran to `stopping_early: wallclock_cap` at step 4544 / 2388s.

| metric | variant | Exp 24 control | Δ |
|---|---|---|---|
| final step | 4544 | 4531 | +13 (matched) |
| training wall | 2388s | 2352s | +1.5% |
| final step val_bpb (pre-EMA) | 1.0900 (est.) | 1.0879 | ~+0.002 |
| **pre-quant post-EMA val_bpb** | **1.08788** | **1.08670** | **+0.00118** |

**Signal gate: NOT MET** on both criteria:

| criterion | required | actual | pass? |
|---|---|---|---|
| variant ≤ Exp 24 at ≥3 of last 4 train_loss milestones (4200/4300/4400/4500) | ≥3/4 | **0/4** (Δ +0.0026 to +0.0059) | ✗ |
| final pre-quant val_bpb | ≤ 1.0847 | 1.08788 | ✗ (miss by +0.00318) |

## Noise/signal judgment — kill

**Not noise.** SOTA intra-seed std is ~0.0002; our Δ is +0.00118 at the final pre-quant — 6× the SOTA std. Direction is also consistent through the entire second half of training.

Train_loss trajectory is **nuanced but directional:**

| step | variant | Exp 24 | Δ |
|---|---|---|---|
| 100 | 4.5047 | 4.4674 | +0.0373 |
| 500 | 3.2753 | 3.2805 | **−0.0052** |
| 1000 | 3.2358 | 3.2346 | +0.0012 |
| 1500 | 3.1259 | 3.1254 | +0.0005 |
| 2000 | 3.0593 | 3.0602 | **−0.0009** |
| 3000 | 2.9885 | 2.9848 | +0.0037 |
| 4000 | 2.8867 | 2.8812 | +0.0055 |
| 4500 | 2.8049 | 2.8023 | +0.0026 |

Three regimes:
1. **Steps 100-300:** variant worse (zero-init BigramHash adaptation burn).
2. **Steps 500-2000:** essentially tied — variant slightly BETTER at step 500 (−0.0052) and step 2000 (−0.0009). *This is the only part that gives "maybe BigramHash is helping" signal*, and it's inside noise.
3. **Steps 2500+:** variant consistently worse by 0.003-0.006 through all of warmdown.

So BigramHash isn't catastrophic — it holds parity with Exp 24 in mid-training — but **it loses during warmdown**. Warmdown is where model quality consolidates for the final submission number; losing there is what matters.

## Why it didn't work (hypotheses — for learning, not action)

1. **April SOTA primitives subsume BigramHash.** Depth recurrence (3 layers, activated at frac=0.35) adds ~6 effective layers of capacity at the layers where bigram-style pattern matching would happen. Parallel residuals from layer 7 give attention+MLP shared input. QK-gain 5.25 sharpens attention distributions. Each of these gives the model more flexibility to encode short-range statistics directly. By the time BigramHash's embedding table has learned useful bigram signals, the model has already encoded those patterns through the richer primitives.

2. **SP8192 tokenizer + 3072 buckets is highly collisional.** Prior submission that reported +0.003-0.005 from BigramHash used SP1024 or earlier. SP8192 has an 8× larger token vocab, so bigram combinatorics are ~64× more varied. 3072 buckets collides aggressively — maybe 20-30× collision rate vs SP1024. The "shortcut" BigramHash provides is diluted by cross-bigram interference in each bucket.

3. **Warmdown interaction.** During warmdown, LR → 0 over the last 72% of training. BigramHash's embedding gets fewer effective updates proportional to its depth-in-stack (gradient attenuation through recurrence + parallel residuals). Exp 24's baseline model has the same attenuation but doesn't depend on the bigram shortcut, so warmdown extracts more from it.

None of these hypotheses are actionable for the record push — fixing BigramHash would require either retokenizing FineWeb (weeks) or scaling up buckets significantly (breaks the 16MB artifact budget). Kill.

## Secondary finding — GPTQ crashes when `BIGRAM_VOCAB_SIZE > 0`

After pre-quant post-EMA eval wrote the number we needed, the training script continued to GPTQ:

```
GPTQ:collecting Hessians from calibration data...
GPTQ:collected 67 Hessians in 13.0s
KeyError: 'bigram.embed.weight'
```

**Root cause:** `gptq_mixed_quantize` (`train_gpt_sota.py:848-862`) treats every state_dict entry >65536 params as a quantization candidate. `bigram.embed.weight` is 3072×112 = 344,064 → qualifies. But `collect_hessians` (`train_gpt_sota.py:763-807`) only hooks `CastedLinear` modules, not `nn.Embedding`. So `hessians['bigram.embed.weight']` doesn't exist → KeyError when the loop reaches it.

The SOTA submission always ran `BIGRAM_VOCAB_SIZE=0`, so this code path was never exercised. It's a latent code bug, not something we introduced.

**Impact on our result: none.** Pre-quant post-EMA val_bpb landed *before* the crash — the signal-gate number is valid. We just didn't get a quantized artifact, which we wouldn't have used for a screening run anyway.

**If BigramHash were ever revived** (unlikely given the kill, but noting): one-line fix. Either:
- Add `bigram.embed.weight` to the passthrough-as-float16 branch in `gptq_mixed_quantize` (alongside the existing `tok_emb` special case).
- Add an `nn.Embedding` hook in `collect_hessians` so bigram gets its own Hessian and quantizes normally.

Option A is simpler; option B preserves int6 quantization of the bigram table and saves ~260KB of artifact space.

## Secondary finding — step-matched comparison held cleanly

Execution worried about step-count handicap: our pod was ~8-10% slower per step during early training (compile overhead), projecting ~4005-step finish vs Exp 24's 4531. By the end, our step rate had caught up (~1.92 steps/sec), and we finished at step 4544 — **13 more steps than Exp 24**. Final comparison IS apples-to-apples at the step level.

This validates the "screen at matched step, don't sweat early-compile throughput" lesson. For future screens on fresh pods, the first ~500 steps of compile-warmup drag amortizes naturally.

## Cost accounting

- Spec estimate: ~$4.00 base.
- Actual: **$5.11** balance drop ($25.37 → $20.26).
- Overshoot ~28% from (a) slightly longer total wall than Exp 24 (pod creation + SSH + rsync + post-training eval tail), and (b) early-training throughput drag on a fresh container disk.
- **Training itself was $4.00**; the $1.11 overshoot was logistics.

Clean budget behavior, no wasted pods.

## Decision: **KILL**

- Idea `research/ideas/bigram-hash.md` should be updated with `Status: ❌ SHELVED 2026-04-19`.
- Don't retest with larger `BIGRAM_VOCAB_SIZE` (breaks 16MB budget) or different `BIGRAM_DIM` (already picked small).
- Don't retest on a tokenizer with lower bigram collision (retokenizing FineWeb is a weeks-long detour).
- Three candidate-space kills in a row (001, 002, 003) — the April SOTA stack is dense. The remaining $180 of budget should pivot away from "port prior submission tricks" toward more speculative architectural changes.

## Strategic implications

**Third consecutive cheap-screen kill.** Post-training pipeline (001 + 002) and embedding-side addition (003) all fail to improve on SOTA's baseline. The remaining bpb headroom is apparently not in places anyone has left plausible footprints — we've now tested the three most-obvious ports from prior near-SOTA submissions, all dead.

**Candidate set for the remaining budget:**

| candidate | expected Δ | cost | risk |
|---|---|---|---|
| Layerwise LR decay | +0.002-0.005 (speculative) | ~$4 screen on 2×H100 | moderate — hyperparam tuning is brittle |
| num_kv_heads=2 (from 4) | +0.001-0.003 from freed param budget | ~$4 mini + 8H run if signal | low risk (saves params, shift to depth) |
| Wider model_dim 512→576 | +0.002-0.006 from more capacity | ~$5 screen | high risk — may break 16MB budget |
| Train-time logit softcap tuning | speculative | ~$4 | low priority |

None of these have the "prior submission claimed +X" provenance that 001/002/003 had. We're in harder territory now.

**Recommendation:** spec 004 = layerwise LR decay screen, OR num_kv_heads=2 as the single structural change with highest expected cost/benefit. Research picks.

## Next steps

1. Update `research/ideas/bigram-hash.md` with SHELVED status + this evaluation's rationale summary.
2. Append row `003` to `experiments.md` (done in the same commit as this file).
3. Draft spec 004 — research picks between layerwise-LR-decay vs num_kv_heads=2 based on which has clearer signal-to-noise.
4. **Budget check:** $11.37 (000) + $1.90 (001) + $2.87 (002) + $5.11 (003) = **$21.25 spent**. **$178 remaining** (of the $200 hard cap). ~10-11 days of runway at current spend pace.

## Artifacts retained

In repo at `runs/003-bigram-hash-screen/`:
- `variant_train.log` — full stdout with the matched-step comparison data + GPTQ crash trace.
- `summary.md` (execution), `notes.md` (execution narrative).

On NA-1 volume: **nothing spec-003-specific retained.** GPTQ crashed before any `.ptz` was written. From-scratch training with no checkpoint policy — nothing to hotstart from anyway (the model was a dead end). Volume is effectively unchanged vs before this run.
