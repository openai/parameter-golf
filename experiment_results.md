# Experiment Results

Baseline reference: PR #549 — val_bpb 1.1194 (3-seed mean, 8xH100, dim=512, 11 layers)

## Experiment 0: Latest baseline

- **Date**: 2026-03-28
- **Hardware**: 4xH100 80GB
- **Steps completed**: 7,392 / 9,000

### Results
| Metric | Value |
|--------|-------|
| Post-EMA val_bpb | 1.1366 |
| Final int6 sliding window val_bpb | 1.1214 |
| **Post-TTT sliding window val_bpb** | **1.1191** |

---

## Experiment 1: MODEL_DIM=576 (all else PR #549 defaults)

- **Date**: 2026-03-24
- **Hardware**: 4xH100 80GB
- **Key changes**: MODEL_DIM=576 (up from 512), MAX_WALLCLOCK_SECONDS=1200
- **Other params**: ITERATIONS=9000, all other hyperparams at code defaults (matching PR #549)
- **model_params**: 33,968,348 (~34M, over budget)
- **Steps completed**: 5,609 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~214ms
- **Peak memory**: 24,301 MiB

### Results
| Metric | Value |
|--------|-------|
| Pre-EMA val_bpb | 1.1284 |
| Post-EMA val_bpb | 1.1277 |
| **Final int6 quantized val_bpb** | **1.1359** |
| Submission size (int6+lzma) | 19,525,669 bytes (~19.5 MB) |

### Notes
- Over param budget (~34M vs typical ~24M), so not submittable as-is.
- Quantization gap is large: 1.1277 -> 1.1359 (+0.0082), likely because bigger model loses more to int6.
- Only got ~5.6k steps due to slower step time at larger dim on 4 GPUs.
- No TTT was run (would need separate eval pass).

---

## Experiment 2: NUM_LAYERS=12 (dim=512, all else PR #549 defaults)

- **Date**: 2026-03-24
- **Hardware**: 4xH100 80GB
- **Key changes**: NUM_LAYERS=12 (up from 11), MAX_WALLCLOCK_SECONDS=1200
- **Other params**: ITERATIONS=9000, MODEL_DIM=512, all other hyperparams at code defaults
- **model_params**: 29,355,620 (~29M, over budget but closer than dim=576)
- **Steps completed**: 6,878 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~174ms
- **Peak memory**: 23,484 MiB

### Results
| Metric | Value |
|--------|-------|
| Pre-EMA val_bpb | 1.1317 |
| Post-EMA val_bpb | 1.1307 |
| Final int6 quantized val_bpb | 1.1390 |
| **Final int6 sliding window val_bpb** | **1.1153** |
| Submission size (int6+lzma) | 17,275,453 bytes (~17.3 MB) |

### Notes
- Also over param budget (~29M) but less so than dim=576.
- Quantization gap: 1.1307 -> 1.1390 (+0.0083), similar to exp 1.
- Sliding window eval (stride 64) brings it to 1.1153 — better than PR #549 baseline (1.1194) pre-TTT.
- Got more steps (6,878 vs 5,609) due to faster per-step time at dim=512.
- No TTT was run.

---

## Experiment 3: NUM_LAYERS=12 + TTT (dim=512, all else PR #549 defaults)

- **Date**: 2026-03-24
- **Hardware**: 4xH100 80GB
- **Key changes**: NUM_LAYERS=12, TTT_ENABLED=1, MAX_WALLCLOCK_SECONDS=1200
- **Other params**: ITERATIONS=9000, MODEL_DIM=512, all other hyperparams at code defaults
- **model_params**: 29,355,620 (~29M)
- **Steps completed**: 6,879 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~174ms
- **TTT time**: 620s (1893 chunks, 3 epochs each)

### Results
| Metric | Value |
|--------|-------|
| Post-EMA val_bpb | 1.1304 |
| Final int6 quantized val_bpb | 1.1387 |
| Final int6 sliding window val_bpb | 1.1151 |
| **Post-TTT sliding window val_bpb** | **1.1126** |

### Notes
- TTT gave a -0.0025 gain over sliding window (1.1151 -> 1.1126), similar to PR #549's TTT gain.
- **Beats PR #549's 1.1194 by 0.0068 BPB** — but still over param budget (~29M).
- TTT took 620s which would be within a 10-min eval constraint.

---

## Experiment 3b: NUM_LAYERS=13 + TTT (dim=512, no recurrence)

- **Date**: 2026-03-27
- **Hardware**: 4xH100 80GB
- **Key changes**: NUM_LAYERS=13 (independent layers, no recurrence), TTT_ENABLED=1, MAX_WALLCLOCK_SECONDS=1200
- **Other params**: ITERATIONS=9000, MODEL_DIM=512, all other hyperparams at code defaults
- **model_params**: 31,651,436 (~32M, well over budget)
- **Steps completed**: 5,995 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~200ms
- **Peak memory**: 25,474 MiB
- **TTT time**: 696s (1893 chunks, 3 epochs each)
- **Submission size**: 18,535,401 bytes (~18.5 MB)

### Results
| Metric | Value |
|--------|-------|
| Post-EMA val_bpb | 1.1297 |
| Final int6 quantized val_bpb | 1.1374 |
| Final int6 sliding window val_bpb | 1.1140 |
| **Post-TTT sliding window val_bpb** | **1.1117** |

### Comparison to 12-layer (Exp 3) and dual recurrence (Exp 5)
| Metric | 12L (Exp 3) | 13L (Exp 3b) | Recur 4,5 (Exp 5) |
|--------|-------------|--------------|---------------------|
| Params | 29.4M | 31.7M | 27.0M |
| Virtual depth | 12 | 13 | 13 |
| Steps completed | 6,879 | 5,995 | 6,389 |
| Step avg | ~174ms | ~200ms | ~188ms |
| Sliding window val_bpb | 1.1151 | 1.1140 | 1.1187 |
| Post-TTT val_bpb | 1.1126 | 1.1117 | 1.1163 |

### Notes
- 13L independent beats 12L by only 0.0009 BPB (1.1117 vs 1.1126) despite +2.3M params and same virtual depth +1.
- Got ~900 fewer steps than 12L due to slower step time (200ms vs 174ms).
- At matched virtual depth of 13: independent 13L (1.1117) beats dual recurrence (1.1163) by 0.0046 — but costs +4.7M params.
- Diminishing returns from adding independent layers: 11→12 gave ~0.007 BPB gain, but 12→13 gives only ~0.001.
- Well over param budget (~32M), not submittable.

---

## Experiment 4: RECUR_LAYER=5 + TTT (depth recurrence, 11 physical → 12 virtual layers)

- **Date**: 2026-03-25
- **Hardware**: 4xH100 80GB
- **Key changes**: RECUR_LAYER=5 (layer 5 duplicated), TTT_ENABLED=1, MAX_WALLCLOCK_SECONDS=1200
- **Other params**: ITERATIONS=9000, MODEL_DIM=512, NUM_LAYERS=11 (physical), all else defaults
- **model_params**: 26,996,324 (~27M) — only ~2.7M over baseline from extra block scalars
- **Steps completed**: 6,884 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~174ms (same as full 12-layer)
- **TTT time**: 622s (untied recurrence before TTT)
- **Submission size**: 15,927,562 bytes (~15.9 MB, well under 16MB budget!)

### Results
| Metric | Value |
|--------|-------|
| Post-EMA val_bpb | 1.1354 |
| Final int6 quantized val_bpb | 1.1440 |
| Final int6 sliding window val_bpb | 1.1205 |
| **Post-TTT sliding window val_bpb** | **1.1180** |

### Comparison to full 12-layer (Exp 3)
| Metric | Full 12L (Exp 3) | Recur L5 (Exp 4) | Delta |
|--------|-----------------|-------------------|-------|
| Params | 29.4M | 27.0M | -2.4M |
| Submission size | 17.3 MB | 15.9 MB | -1.4 MB |
| Sliding window val_bpb | 1.1151 | 1.1205 | +0.0054 |
| Post-TTT val_bpb | 1.1126 | 1.1180 | +0.0054 |

### Notes
- Recurrence adds depth for free in compute, but shared weights cost ~0.005 BPB vs independent layers.
- TTT untying gave a -0.0025 gain (1.1205 -> 1.1180), same magnitude as independent layers.
- Submission size is much smaller (15.9 MB vs 17.3 MB) since banks stay at 11-layer size.
- Still beats PR #549 baseline (1.1194) by 0.0014 BPB, with a smaller model.

---

## Experiment 4b: Tied TTT (same checkpoint as Exp 4, no untying)

- **Post-TTT val_bpb**: **1.1179** (vs 1.1180 untied — negligible difference)
- Conclusion: untying doesn't help with 3-epoch TTT. Tied is fine.

---

## Experiment 5: RECUR_LAYERS=4,5 + tied TTT (dual recurrence, 11 physical → 13 virtual layers)

- **Date**: 2026-03-25
- **Hardware**: 4xH100 80GB
- **Key changes**: RECUR_LAYERS=4,5 (4,5,4,5 pattern), TTT_ENABLED=1, TTT_UNTIE=0
- **Other params**: ITERATIONS=9000, MODEL_DIM=512, NUM_LAYERS=11 (physical), all else defaults
- **model_params**: 26,998,380 (~27M)
- **Steps completed**: 6,389 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~188ms (up from 174ms with single recurrence — extra virtual layer costs ~14ms/step)
- **TTT time**: 655s (tied)
- **Submission size**: 15,944,748 bytes (~15.9 MB)

### Results
| Metric | Value |
|--------|-------|
| Post-EMA val_bpb | 1.1337 |
| Final int6 quantized val_bpb | 1.1421 |
| Final int6 sliding window val_bpb | 1.1187 |
| **Post-TTT sliding window val_bpb** | **1.1163** |

### Full comparison
| | PR #549 | Recur L5 (Exp 4) | Recur L4,5 (Exp 5) | Full 12L (Exp 3) |
|---|---|---|---|---|
| Virtual depth | 11 | 12 | **13** | 12 |
| Params | ~24M | ~27M | ~27M | ~29M |
| Submission size | ~19.5 MB | 15.9 MB | 15.9 MB | 17.3 MB |
| Steps completed | ~7,180 | 6,884 | 6,389 | 6,879 |
| Post-TTT val_bpb | 1.1194 | 1.1179 | **1.1163** | **1.1126** |

### Notes
- Dual recurrence (1.1163) beats single recurrence (1.1179) by 0.0016 BPB.
- Beats PR #549 by 0.0031 BPB, with ~27M params and ~16 MB submission.
- Gap to full independent 12-layer (1.1126) is 0.0037 — weight sharing costs more with 2 repeated layers.
- Step time increased to ~188ms (from 174ms), resulting in ~500 fewer steps in the wallclock budget.
- The extra virtual depth helps despite fewer training steps.

---

## Experiment 6: RECUR_LAYERS=4,5,6 + tied TTT (triple recurrence, 11 physical → 14 virtual layers)

- **Date**: 2026-03-25
- **Hardware**: 4xH100 80GB
- **Key changes**: RECUR_LAYERS=4,5,6, TTT_ENABLED=1, TTT_UNTIE=0
- **model_params**: 27,000,948 (~27M)
- **Steps completed**: 5,977 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~201ms
- **TTT time**: 700s (tied)
- **Submission size**: 15,921,244 bytes (~15.9 MB)

### Results
| Metric | Value |
|--------|-------|
| Post-EMA val_bpb | 1.1365 |
| Final int6 sliding window val_bpb | 1.1217 |
| **Post-TTT sliding window val_bpb** | **1.1190** |

### Notes
- Triple recurrence (1.1190) is worse than dual (1.1163) despite more virtual depth (14 vs 13).
- Lost ~400 steps vs dual due to slower step time (201ms vs 188ms).
- Per-step learning was slightly better, but not enough to overcome fewer total steps.
- **Conclusion: dual recurrence at layers 4,5 is the sweet spot for this wallclock budget.**

---

## Experiment 5b: Untied TTT on dual recurrence (RECUR_LAYERS=4,5)

- **Date**: 2026-03-25
- **Same checkpoint as Exp 5**, only TTT_UNTIE=1 differs
- **Post-TTT sliding window val_bpb**: **1.1163** (identical to tied 1.1163)
- **Conclusion**: Untying doesn't help for dual recurrence either, matching the single-recurrence finding (Exp 4b). Tied TTT is sufficient.

---

## Experiment 7: Eval-time bottleneck (RECUR_LAYERS=4,5, EVAL_BOTTLENECK_REPS=1)

- **Date**: 2026-03-25
- **Hardware**: 4xH100 80GB
- **Key changes**: EVAL_ONLY=1, EVAL_BOTTLENECK_REPS=1 (adds 2 extra virtual layers between encoder/decoder at eval time)
- **Same checkpoint as Exp 5** (RECUR_LAYERS=4,5 trained model)
- **Virtual depth at eval**: 15 (13 trained + 2 bottleneck)
- **TTT time**: 724s (tied)

### Results
| Metric | Value |
|--------|-------|
| Final int6 sliding window val_bpb (with bottleneck) | 1.1472 |
| **Post-TTT sliding window val_bpb** | **1.1219** |

### Comparison to no-bottleneck (Exp 5)
| Metric | No bottleneck (Exp 5) | +1 bottleneck rep (Exp 7) | Delta |
|--------|----------------------|--------------------------|-------|
| Pre-TTT sliding window | 1.1187 | 1.1472 | +0.0285 |
| Post-TTT sliding window | 1.1163 | 1.1219 | +0.0056 |

### Notes
- The extra untrained bottleneck blocks start far from useful (1.1472 vs 1.1187 pre-TTT).
- TTT recovers most of the gap but not all — final result is 0.0056 worse than without bottleneck.
- **Conclusion: Eval-time bottleneck with deepcopy of trained blocks does not help.** The copied blocks need to be trained to integrate into the forward pass. 3-epoch TTT is insufficient to close the gap.
- Possible improvement: initialize bottleneck blocks as identity/near-identity rather than copies of mid-network blocks.

---

## Experiment 8: Dual recurrence layer placement sweep (no TTT)

- **Date**: 2026-03-26
- **Hardware**: 4xH100 80GB
- **Key changes**: Swept RECUR_LAYERS across 6 pairs to find optimal placement, no TTT
- **Other params**: ITERATIONS=9000, MODEL_DIM=512, NUM_LAYERS=11 (physical), all else defaults
- **model_params**: 26,998,380 (~27M) for all runs
- **Virtual depth**: 13 (11 physical + 2 recurrent) for all runs
- **Step avg**: ~189ms across all runs
- **Wallclock**: 1200s (all runs hit cap)

### Results

| RECUR_LAYERS | Steps | Post-EMA val_bpb | Int6 val_bpb | Int6 sliding window val_bpb | Submission size |
|--------------|-------|-------------------|--------------|----------------------------|-----------------|
| 0,1 | 6,309 | 1.1396 | 1.1479 | 1.1245 | 15.92 MB |
| 2,3 | 6,332 | 1.1348 | 1.1431 | 1.1196 | 15.94 MB |
| 3,4 | 6,337 | 1.1346 | 1.1434 | 1.1200 | 15.95 MB |
| **4,5** | **6,335** | **1.1340** | **1.1424** | **1.1190** | **15.94 MB** |
| 5,6 | 6,330 | 1.1358 | 1.1445 | 1.1211 | 15.95 MB |
| 6,7 | 6,335 | 1.1361 | 1.1444 | 1.1208 | 15.95 MB |
| 8,9 | 6,340 | 1.1414 | 1.1498 | 1.1263 | 16.03 MB |
| 9,10 | 6,342 | 1.1419 | — (incomplete) | — (incomplete) | 15.95 MB |

### Notes
- **Best placement: layers 4,5** (1.1190 sliding window), confirmed even with finer-grained sweep.
- Clear U-shaped curve centered on 4,5, with performance degrading toward both ends.
- The immediate neighbors (3,4) at 1.1200 and (5,6) at 1.1211 are both worse — 4,5 is a sharp optimum, not a broad plateau.
- Early layers (0,1) are 0.0055 worse than (4,5); late layers (8,9) are 0.0073 worse.
- Notably (5,6) at 1.1211 is slightly worse than (6,7) at 1.1208 — the curve isn't perfectly smooth on the right side.
- The 4,5 sweet spot coincides with the U-Net encoder/decoder boundary (virtual layers 6/7 out of 13), placing recurrence exactly at the skip connection hinge point.
- No TTT was run in this sweep; adding TTT to the (4,5) winner matches Exp 5's result of 1.1163.
- The recur_9_10 run had an incomplete evaluation (log truncated after submission size).

---

## Experiment 9: Delayed recurrence (RECUR_LAYERS=4,5 activated at step 3000)

- **Date**: 2026-03-27
- **Hardware**: 4xH100 80GB
- **Key changes**: RECUR_LAYERS=4,5, RECUR_START_STEP=3000 (train as 11L for 3k steps, then activate recurrence), TTT_ENABLED=1, TTT_UNTIE=0
- **Other params**: ITERATIONS=9000, MODEL_DIM=512, NUM_LAYERS=11 (physical), all else defaults
- **model_params**: 26,998,380 (~27M)
- **Steps completed**: 6,345 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~161ms (steps 0-3000), ~189ms (steps 3000-6345), blended ~189ms
- **Peak memory**: 25,386 MiB
- **Submission size**: 16,094,569 bytes (~16.1 MB)

### Results
| Metric | Value |
|--------|-------|
| Post-EMA val_bpb | 1.1347 |
| Final int6 quantized val_bpb | 1.1437 |
| Final int6 sliding window val_bpb | 1.1201 |

### Comparison to always-on recurrence (Exp 5)
| Metric | Always-on (Exp 5) | Delayed 3k (Exp 9) | Delta |
|--------|-------------------|---------------------|-------|
| Steps completed | 6,389 | 6,345 | -44 |
| Pre-3k step avg | 188ms | 161ms | -27ms |
| Post-3k step avg | 188ms | 189ms | same |
| Sliding window val_bpb | 1.1187 | 1.1201 | +0.0014 (worse) |

### Notes
- **Delayed recurrence is worse (1.1201 vs 1.1187)** despite saving ~80s of wallclock in the first 3k steps.
- Only gained ~44 extra steps total — the fast 161ms phase saves 27ms/step × 3000 = ~80s, but that's only ~420 extra steps at 189ms, and the average over the full run is nearly the same.
- The core problem: the shared weights weren't trained for dual-use during the critical first 3k steps. When recurrence activates, the recurrence blocks' scalars (attn_scale, mlp_scale, resid_mix) are at init values and the bank weights haven't learned to handle two passes. 3345 steps of recurrence training isn't enough to catch up.
- **Conclusion (revised in Exp 9b):** The poor result was due to `torch._dynamo.reset()` at step 3000 causing a ~90s recompilation, eating all the wallclock savings and then some.

---

## Experiment 9b: Delayed recurrence with pre-warmed compilation

- **Date**: 2026-03-27
- **Hardware**: 4xH100 80GB
- **Key changes**: Same as Exp 9, but pre-warm both torch.compile traces during warmup phase to avoid recompilation cost at step 3000
- **Other params**: ITERATIONS=9000, RECUR_LAYERS=4,5, RECUR_START_STEP=3000, TTT_ENABLED=1, TTT_UNTIE=0
- **model_params**: 26,998,380 (~27M)
- **Steps completed**: 6,747 / 9000 (wallclock capped at 1200s)
- **Step avg**: ~178ms blended (161ms pre-recurrence, ~189ms post-recurrence)
- **Peak memory**: 25,385 MiB
- **TTT time**: 687s
- **Submission size**: 16,055,875 bytes (~16.1 MB)

### Results
| Metric | Value |
|--------|-------|
| Post-EMA val_bpb | 1.1329 |
| Final int6 quantized val_bpb | 1.1414 |
| Final int6 sliding window val_bpb | 1.1179 |
| **Post-TTT sliding window val_bpb** | **1.1153** |

### Comparison
| Metric | Always-on (Exp 5) | Delayed no-prewarm (Exp 9) | Delayed prewarmed (Exp 9b) |
|--------|-------------------|---------------------------|---------------------------|
| Steps completed | 6,389 | 6,345 | **6,747** |
| Avg step time | 188ms | 189ms | **178ms** |
| Sliding window val_bpb | 1.1187 | 1.1201 | **1.1179** |
| Post-TTT val_bpb | 1.1163 | — (crashed) | **1.1153** |

### Notes
- Pre-warming the torch.compile trace during warmup recovered the ~390 extra steps that recompilation stole in Exp 9.
- **Beats always-on recurrence (1.1163) by 0.0010 BPB** with the same param count.
- The 358 extra steps from the fast 161ms/step pre-recurrence phase more than compensate for the co-adaptation gap.
- New best recurrence result: **1.1153 post-TTT** at 27M params.
- Beats PR #549 baseline (1.1194) by **0.0041 BPB**.

---

## Experiment 10: RECUR_START_STEP sweep (3000, 3500, 4000)

- **Date**: 2026-03-27
- **Hardware**: 4xH100 80GB
- **Key changes**: Swept RECUR_START_STEP with pre-warmed compilation, RECUR_LAYERS=4,5, TTT_ENABLED=1, TTT_UNTIE=0
- **model_params**: 26,998,380 (~27M) for all runs

### Results

| RECUR_START_STEP | Steps | Blended avg | Sliding window val_bpb | Post-TTT val_bpb |
|------------------|-------|-------------|------------------------|------------------|
| 0 (Exp 5) | 6,389 | 188ms | 1.1187 | 1.1163 |
| 2500 | 6,668 | 180ms | 1.1181 | 1.1154 |
| **3000 (Exp 9b)** | **6,747** | **178ms** | **1.1179** | **1.1153** |
| 3500 | 6,828 | 176ms | 1.1177 | 1.1154 |
| 4000 | 6,895 | 174ms | 1.1180 | 1.1156 |

### Notes
- **Remarkably flat from 2500-3500** — all within 0.0001 of each other (1.1153-1.1154 post-TTT).
- 3000 is marginally best at 1.1153 but practically indistinguishable from 2500 and 3500.
- 4000 shows the first clear degradation (1.1156) — too little recurrence training time.
- **Conclusion: RECUR_START_STEP=3000 is the sweet spot.** The flat region 2500-3500 suggests the model is robust to this parameter, but 3000 gives the best balance of step budget vs recurrence training time.

---

## Experiment 11: WARMDOWN_ITERS sweep with delayed recurrence (RECUR_START_STEP=3000)

- **Date**: 2026-03-27
- **Hardware**: 4xH100 80GB
- **Key changes**: Swept WARMDOWN_ITERS to give recurrence blocks more full-LR training time
- **Other params**: RECUR_LAYERS=4,5, RECUR_START_STEP=3000, TTT_ENABLED=1, TTT_UNTIE=0

### Results

| WARMDOWN_ITERS | Steps | Post-EMA val_bpb | Sliding window val_bpb | Post-TTT val_bpb |
|----------------|-------|------------------|------------------------|------------------|
| 1000 | 6,741 | 1.1450 | 1.1258 | 1.1232 |
| 2000 | 6,742 | 1.1367 | 1.1192 | 1.1171 |
| **3000** | **6,746** | **1.1335** | **1.1174** | **1.1151** |
| 3500 (default) | 6,747 | 1.1329 | 1.1179 | 1.1153 |

### Notes
- **WARMDOWN_ITERS=3000 is new best: 1.1151 post-TTT**, beating default 3500 by 0.0002.
- Clear monotonic trend from 1000→3000, then plateaus at 3000-3500.
- 3000 gives recurrence ~750 steps at full LR (vs ~240 with 3500) while still providing enough cooldown.
- Below 2000 the model fails to converge — the cooldown is too short.
- **New best overall: RECUR_START_STEP=3000 + WARMDOWN_ITERS=3000 → 1.1151 post-TTT at 27M params.**
- Beats PR #549 baseline (1.1194) by **0.0043 BPB**.
