# Parameter Golf -- Comprehensive Findings Document
**Team: Frosty40 / Farnsworth Tech | Competition: March 18 -- April 30, 2026**
**Last updated: 2026-03-25**

---

## Current SOTA

- **PR #753: 0.9625 mean BPB** (seeds 42=0.9631, 2045=0.9620, 7=0.9624)
- Architecture: 11L/512d U-Net, LeakyReLU-squared slope 0.5, XSA last 4, BigramHash 1536, ROPE 24
- N-gram: 7-gram backoff orders 2-7, entropy-adaptive alpha (0.05-0.60), center 4.0, scale 2.0, min_count 2, 4M buckets
- Artifact: ~15.6MB int6+zstd
- SOTA file hash: 147bbccc (96,116 bytes)
- Source: `concepts/podracer/sota/run.sh` + `concepts/podracer/sota/train_gpt.py`

## NEW RECORD: Cubric Lite (pending multi-seed)

- **0.9362 BPB** (seed 2045, single seed) — **0.026 better than PR #753**
- Same architecture, same training, same n-gram tables
- Only change: cubric lite per-order adaptive alpha scaling (CUBRIC_CADENCE=32)
- Converged multipliers: `o2:0.300 o3:0.300 o4:0.970 o5:2.000 o6:2.000 o7:2.000`
- **Key insight: orders 2-3 were actively hurting BPB.** Suppressing their alpha to 30% of base and boosting orders 5-7 to 200% (capped at alpha_max) = 0.026 BPB gain
- Sliding BPB (no n-gram): 1.1199 — identical to baseline, confirming model unchanged
- REQUIRES: zstd compression (zlib produces 17MB, zstd ~15.7MB), multi-seed verification
- Source: `concepts/podracer/podracer_green/run.sh` + `concepts/podracer/podracer_green/train_gpt.py`
- **Original contribution: per-order adaptive alpha scaling on score-first n-gram backoff**

### SOTA Seed Breakdown (with n-gram)

| Seed | Sliding BPB (no n-gram) | 7-gram Backoff BPB | Artifact | N-gram Config |
|------|-------------------------|-------------------|----------|---------------|
| 1337 | 1.1195 | 1.0217 | 15.59 MB | **order=5, alpha=0.2** (OLD CONFIG -- outlier) |
| 42 | 1.1210 | **0.9631** | 15.59 MB | order=7, alpha=0.3 (correct) |
| 2045 | 1.1196 | **0.9620** | 15.71 MB | order=7, alpha=0.3 (correct) |
| 7 | -- | **0.9624** | -- | order=7, alpha=0.3 (correct) |
| **Mean (42/2045/7)** | **1.1200** | **0.9625** | -- | -- |

### Seed 1337 Outlier Explained

Seed 1337 ran with the **old Podracing I config** (order=5, alpha=0.2) instead of the Podracing II config (order=7, alpha=0.3). This is confirmed in the training log: `ngram_eval:order=5 alpha=0.2` vs seeds 42/2045 which show `ngram_eval:order=7 alpha=0.3`. The 0.06 BPB gap (1.0217 vs ~0.962) is entirely due to the n-gram configuration, not the neural model. The sliding BPB without n-gram is comparable across all seeds (1.1195-1.1210).

---

## Proven Findings (backed by data)

### Architecture Findings

1. **Weight sharing + wider layers is the dominant fractal effect.** Fractal-only (3x3, 864d) beats 9-unique-layer baseline (512d) by 7.1% BPB (2.5953 vs 2.7927) with fewer parameters. The width from sharing is the value, not the recurrence. Source: `RESULTS.md`, DGX Spark 300-step experiments.

2. **MLP 4x is a massive quality lever (+2% relative BPB over 3x).** But 12 unique layers with MLP 4x blows the 16MB budget. Weight sharing enables MLP 4x. Source: `records/track_10min_16mb/2026-03-23_Frugendorff_Squared_6x2_640d_MLP4/README.md`, Qwen overnight sweep.

3. **Asymmetric sharing (4 flat + 2 shared) beats symmetric sharing (6x2) by 0.010 BPB** (1.1375 vs 1.1478). More unique parameters + small shared tail is strictly better than balanced sharing. Source: `MICRO_CRAWLER_RESULTS.md`.

4. **11L/512d U-Net is the strongest frame.** 11 layers, 512 dim, 8 heads, 4 KV heads (GQA 2:1), head_dim=64. 5 encoder + 6 decoder with skip connections. Beats all fractal/crawler variants on sliding BPB in wallclock-limited setting. Source: all GS v7 results.

5. **LeakyReLU-squared (slope 0.5) improves over standard ReLU-squared.** F1 Legal LB profile with leaky_relu_sq gave 1.1195 (seed 1337) vs PR #587 baseline 1.1203. -0.0008 BPB. Source: `concepts/f1/RESULTS.md`.

6. **XSA last 4 is the sweet spot.** XSA on all 11 layers gives -0.0006 BPB improvement but artifact is 400KB bigger (16.02MB, over limit by 24KB). XSA-4 stays under budget. Source: session state memory, XSA-11 experiments.

7. **BigramHash 1536 vs 2048:** Smaller bigram vocab saves ~400KB artifact size while being quality-neutral. Enables size headroom for other features. Source: `concepts/f1/RESULTS.md`, F1 Legal LB.

8. **12L/480d gives head_dim=30 (invalid for FA3).** Must use 512d/16H (head_dim=32) for FlashAttention 3 compatibility. Source: `records/leapfrog_results_20260322.md`.

### Quantization Findings

9. **GPTQ is the single biggest post-training improvement: -0.0027 BPB.** Hessian-aware error compensation reduces quant tax from 0.0082 to 0.0058 BPB. Column reordering by ascending Hessian diagonal, block-128, percdamp=0.01, 256 calibration samples. All 66 layers calibrated via GPTQ (0 naive fallback). Source: `records/track_10min_16mb/2026-03-23_11L_GPTQ_TTT_EMA_QAT_1.1206/README.md`.

10. **Quant gap scales with double-fire frequency: 5x reduction from cad1 to cad4.** cad1: 0.136, cad2: 0.081, cad3: 0.061, cad4: 0.059 (4x2 architecture). For 6x2: cad1: 0.196, cad4: 0.066. Heavy reuse creates multi-modal weight distributions with outliers that break fixed-point quantization. Source: `experiments/H1_cadence_characterization/HYPOTHESIS.md`, `experiments/H2_cadence_x_architecture/HYPOTHESIS.md`.

11. **EMA instability from parameter reuse.** EMA gap scales with reuse frequency: 0.105 BPB at cad1 (all double-fire) vs 0.053 at cad4 (25% double-fire). Any weight-shared/tied architecture will suffer EMA tracking degradation proportional to reuse frequency. Source: `FRUGENDORFF_PR_DRAFT.md`.

12. **zlib vs zstd matters for size (1.3MB difference), not BPB.** Same quantization, different compression. zstd-22 saves ~1.3MB over zlib. Source: `records/leapfrog_results_20260322.md`.

13. **QAT percentile clip mismatch fix = no gain.** Changing QAT STE from row_max to 0.9995 percentile didn't improve quant tax. Source: `records/leapfrog_results_20260322.md`.

14. **15 GPTQ percentiles = no gain over 5.** The original 5 percentiles already find near-optimal clips. Source: `records/leapfrog_results_20260322.md`.

### TTT Findings

15. **TTT burst before EMA works, but only barely (+0.0001 BPB).** Replaying 100 recent batches for 2 epochs at 10% LR, then applying EMA. Source: `records/leapfrog_results_20260322.md`.

16. **Self-distillation = TTT burst = same ceiling. Do not stack.** Using EMA as teacher with KL+CE lands in the same spot as TTT burst. Both techniques capture the same signal, stacking adds nothing. Source: `records/leapfrog_results_20260322.md`.

17. **EMA-first then burst is worse.** Burst must happen before EMA so EMA can smooth the sharpened weights. Source: `records/leapfrog_results_20260322.md`.

18. **EMA-SWA blend (80/20) hurts -- dilutes EMA signal.** Pure EMA is better than blending with SWA. Source: `records/leapfrog_results_20260322.md`.

19. **Short TTT (50 chunks, no EMA) = net neutral.** Chunk-51 peak 1.1104 but distribution shift in chunks 100-400 drags average back to baseline. TTT adds +0.0000 to -0.0001. Source: session state memory.

20. **Model true capacity is 1.1107 BPB** (running average at TTT chunk 51). Individual chunk scores near 50 are ~1.08-1.09. The gap to final score (1.1206) is 0.0099 BPB, which is 8x the margin needed to beat SOTA. Source: project memory `project_1111_target.md`.

21. **AdamW TTT catastrophic on relu-squared architecture.** seed 1337: 1.1498 BPB (200 chunks). Short window (50 chunks): 1.1248, still worse than SGD. SwiGLU architecture handles AdamW TTT well (1.0763). Architecture is the multiplier for AdamW TTT. Source: `records/leapfrog_results_20260322.md`.

22. **TTT is now banned for submissions** (competition rules update, issue #402). All TTT results are historical only. Score-first protocol is the only legal approach. Source: `feedback_illegal_ttt.md`.

### Training Findings

23. **train_seq_len=1024 is catastrophic.** Only 6% more steps but massive quality loss (1.2224 vs 1.1232). Partial RoPE extrapolation from 1024 to 2048 is insufficient. Source: `records/leapfrog_results_20260322.md`.

24. **Warmdown fix HURT quality.** ITERATIONS=7500 (proper warmdown): 1.1215. ITERATIONS=20000 (no warmdown, high LR to wallclock stop): 1.1201. High LR until wallclock stop + EMA is BETTER than proper convergence. Source: session state memory.

25. **Bigger batch hurts in wallclock-limited training.** 1.5x tokens/step hurt Frugendorff -- fewer total steps offset richer gradients (1.2186 vs 1.2113). Source: `RESULTS.md`.

26. **Single GPU Muon doesn't work.** Plateaued at 1.40 BPB after 20K steps. Muon needs distributed all-reduce for proper operation. Single GPU with gradient accumulation is not equivalent. Source: `RESULTS.md`.

27. **Gravity (auxiliary losses at each loop) hurts at low step counts.** At 300 steps, gravity adds noise. Model learned to turn off early loop gravity: weights [0.13, 0.13, 0.70]. Source: `RESULTS.md`.

### N-gram Findings

28. **7-gram backoff (orders 2-7) with entropy-adaptive alpha is the breakthrough eval technique.** Reduces BPB from ~1.12 to ~0.96 -- a 0.16 BPB improvement from eval-time n-gram interpolation alone. Score-first, backward-looking (cache built from already-scored tokens only). Alpha depends solely on model's own softmax entropy. Source: `records/track_10min_16mb/2026-03-25_PodracingII_backoff7gram_8xH100/README.md`.

29. **N-gram order and alpha are the dominant knobs.** order=5/alpha=0.2 gives 1.0217, order=7/alpha=0.3 gives 0.962x. The 0.06 BPB gap between these configs dwarfs all architecture improvements. Source: training logs in Podracing II record.

30. **N-gram eval is legal.** Cache built from already-scored tokens only. Alpha adjustment depends on model output + past n-gram performance, never future targets. No oracle selection. Source: `records/track_10min_16mb/2026-03-25_PodracingII_backoff7gram_8xH100/README.md`.

### Cadence / Recursion Findings

31. **C-step double-firing provides ZERO measurable benefit.** cad0 (no C-steps) beats all cadence configurations. At full scale: cad0 1.1325 vs cad2 1.1355, with 11% more steps, 31% less memory, and lower quant gap. Source: `experiments/H1_cadence_characterization/HYPOTHESIS.md`.

32. **Less recursion is monotonically better (no U-shape).** At 0.25 scale across all cadences for both 4x2 and 6x2 architectures. val@500 identical for 4x2 across cadences -- C-steps are neutral per step, just cost compute. Source: `experiments/H1_cadence_characterization/HYPOTHESIS.md`.

33. **6x2 is ALWAYS worse than 4x2 at matched cadence.** More crawler blocks = more gradient interference. 6x2 is more cadence-sensitive: val@500 varies by 0.006 across cadences (vs 0.0004 for 4x2). Source: `experiments/H2_cadence_x_architecture/HYPOTHESIS.md`.

34. **6x2 cad1 went BACKWARDS after step 500** (1.3876 -> 1.4059). Gradient interference across 3 crawler blocks with all-C was actively destructive. Source: `experiments/H2_cadence_x_architecture/HYPOTHESIS.md`.

35. **The architecture's value comes from: weight sharing, trigram embedding, XSA, VE injection, GPTQ, SWA, TTT burst, self-distillation -- NOT from recursive refinement.** Source: cadence ablation campaign conclusion.

### Deliberation Gate Findings

36. **Persistent Deliberation needs bidirectional gradient flow.** consensus_ref must be an nn.Parameter (not a detached buffer) so gradients flow BOTH in (loss -> ref) and out (ref -> crawler blocks). Detached EMA consensus goes stale. Source: `project_bidirectional_pd_discovery.md`.

37. **Gate on C-steps only HURT by 0.006 BPB** (Run 3). Gate only trained on 20% of steps -- not enough training signal. Source: `MICRO_CRAWLER_RESULTS.md`.

38. **PD gate on all steps: neutral pre-quant (-0.002), GPTQ recovered.** PD was 0.007 BPB ahead mid-training (steps 5000-7000) but post-processing (EMA/distill) didn't capture the lead. Source: `MICRO_CRAWLER_RESULTS.md`.

39. **PD + cadence are coupled -- detached EMA goes stale with tapered cadence.** Fixed cadence 2 keeps the ref fresh. Source: `MICRO_CRAWLER_RESULTS.md`.

### Crawler Bank Findings

40. **Crawler bank at U-Net bottleneck: per-step learning IS better (+0.016 BPP at step 1500) but net worse (-0.023 sliding BPB).** 15% slower per step -> 14% fewer steps. Post-EMA 0.020 worse. Quant 0.023 worse. In wallclock-limited training, steps beat tricks. Source: `experiments/H4_crawler_bank_on_unet/HYPOTHESIS.md`.

41. **Crawler bank artifact is 0.46MB smaller** (weight sharing compresses well). Only advantage; doesn't help when BPB is worse. Source: `experiments/H4_crawler_bank_on_unet/HYPOTHESIS.md`.

### Other Experiment Findings

42. **MTP (Multi-Token Prediction) HURT: 1.1619 vs 1.1301 baseline.** MTP added 1M params excluded at export. TTT v1 made it worse. Source: `records/exp_a_mtp_20260322.md`.

43. **SwiGLU alone didn't help enough: 1.1348 sliding vs 1.1301 baseline.** TTT v1 hurt SwiGLU too (1.1471 -> 1.1570 roundtrip). Source: `records/exp_b_swiglu_20260322.md`.

44. **Vocab 1536 experiment could not run** (48GB docs needed, only 36GB free). Source: `records/exp_c_vocab1536_20260322.md`.

45. **SwiGLU + AdamW TTT = 1.0763 BPB but 19.6MB (over limit).** GPTQ+OptRot inflates artifact. Architecture is the multiplier for AdamW TTT. Source: `records/leapfrog_results_20260322.md`.

46. **TrigramHash = marginal at best on strong baseline.** 3-token n-gram embeddings added params and overhead without measurable BPB gain. Source: `records/leapfrog_results_20260322.md`.

47. **XSA=3 is too slow: 125.78ms/step (vs ~100ms).** Only 4771/9000 steps, undertrained model, TTT couldn't recover. 1.1797 sliding. Source: `records/v2_tttonly_xsa3_20260322.md`.

48. **TTT v2 (cosine decay + discriminative LR) = worse than baseline.** 1.1315 sliding vs 1.1301 baseline. Temp scaling had no effect (T=1.000). Source: `records/v2_ttt_noXSA_20260322.md`.

49. **12L/4KV/2.625xMLP: faster per step (83.7ms) but worse pre-quant (1.1429 vs 1.1412).** More layers doesn't help when quality per layer drops. Source: `pr374_depth/RESULTS.md`.

50. **Fractal weight sharing at small scale (6Lx2, 512d, 4xMLP) is a dead end.** 18.3M params, 126ms/step, only 4757 steps. Double forward pass costs more compute than it saves in params. 1.1757 sliding, nowhere near 1.1232. Source: `records/leapfrog_results_20260322.md`.

### Autoresearch / Overnight Sweep Findings

51. **Qwen overnight sweep (141 runs, DGX Spark):** Best config: 2 layers x 4 loops, cadence 3 (F/N/N), lr=2e-3, clip=5.0, MLP 3 -> 2.3332 BPB (vs 2.6371 baseline, 12% improvement). Source: `RESULTS.md`.

52. **Frugendorff v2 autoresearch (50+ runs):** Best: 6x1 flat MLP 4x at 2.196 BPB. 4x3 configs also strong (~2.205). Cadence 3 consistently better than cadence 1 or 2. 5x2 sweet spot around 2.23. Source: `autoresearch_frug2_results.csv`.

53. **576plus autoresearch (edge experiments): all 12 runs timed out.** int5 quantization, mixed quant, various GPTQ settings -- all hit the 2400s timeout. No usable results. Source: `autoresearch_576plus_results.csv`.

---

## Active Hypotheses

### CONFIRMED: Cubric Lite — Per-Order Adaptive Alpha (0.026 BPB gain)
- **Status: CONFIRMED on seed 2045. Needs multi-seed.**
- Orders 2-3 suppress to 0.3x alpha (they hurt). Orders 5-7 boost to 2.0x (capped at alpha_max).
- Zero cost: no extra params, no model size change, ~100ms eval overhead.
- Original contribution. No one else in competition has this.
- Next: run seeds 42, 7, 1337 to get 3-seed mean. Install zstd. Submit.

### N-gram Parameter Sweep (pending — vast.ai or RunPod)
- **alpha_max higher (0.70+):** Expected: +0.002-0.010 BPB. May interact with cubric (cubric already effectively raises alpha for good orders).
- **entropy_center lower (3.0):** Expected: +0.001-0.005 BPB. More tokens get high alpha = more tokens where cubric order-scaling matters.
- **buckets 8M (vs 4M):** Expected: +0.001-0.003 BPB. Free lunch.
- **min_count = 1 (vs 2):** Expected: marginal, high risk of noise.
- **order 8+:** Expected: diminishing returns past order 7.
- Source: `concepts/podracer/podracer_red/HYPOTHESIS.md`, `concepts/podracer/podracer_purple/run.sh`.

### Cubric Lite (per-order adaptive alpha scaling)
- Periodically evaluate which n-gram orders are actually helping, then scale alpha per-order.
- Legal: only reads already-scored tokens.
- Expected: +0.001-0.005 BPB. Source: `concepts/cubric_ngram/README.md`, `concepts/cubric_garage/HYPOTHESES.md`.

### Cubric Skiptrace (H5)
- Periodic crawler bank firing + decaying cached delta injection (~1.5% overhead).
- Expected: between control and every-step bank on quality, but closer to control on step count.
- BLOCKED on torch.compile + FA incompatibility on Vast.ai. Ready on RunPod.
- Source: `experiments/H5_cubric_signal/HYPOTHESIS.md`.

### Per-Block Cadence (H3)
- Each crawler block gets its own C/N ratio. Test funnel, diamond, inverse funnel shapes.
- DEPRIORITIZED -- recursion itself found to be net negative.
- Source: `experiments/H3_cadence_gradient_shape/HYPOTHESIS.md`.

### Trigram vs Bigram on SOTA (H6)
- Trigram hash embedding on the 1.1190 model. Expected: +0.001-0.003 BPB.
- Needs code change to make BigramHash configurable.
- Source: `experiments/H6_trigram_on_sota/HYPOTHESIS.md`.

### Weight Sharing Isolation (H8)
- Does weight-shared depth improve BPB over equivalent unique layers, independent of recursion?
- 8 unique flat vs 6 unique + 1 shared x 2. Same effective depth.
- Needs code change.
- Source: `experiments/H8_weight_sharing_isolation/HYPOTHESIS.md`.

### Noisy QAT + Skiptrace (H7)
- Fix crawler bank quant gap using Noisy QAT from PR #363.
- BLOCKED on H5 results.
- Source: `experiments/H7_noisy_qat_skiptrace/HYPOTHESIS.md`.

---

## Dead Ends (confirmed not worth pursuing)

1. **Recursive cadence (C-step double-firing):** Zero benefit at any cadence, any architecture. Pure overhead. Kill it.
2. **MTP (Multi-Token Prediction):** -0.032 BPB worse than baseline. Not viable at this step count.
3. **Fractal weight sharing at 512d scale (6Lx2):** 126ms/step, 4757 steps, 1.1757 BPB. Dead.
4. **TTT v1 (batch, non-score-first):** Now illegal. Also hurt roundtrip BPB consistently.
5. **TTT v2 (cosine decay + discriminative LR):** No improvement over baseline.
6. **EMA-SWA blend:** Dilutes EMA signal. Pure EMA wins.
7. **Stacking burst + distill:** Same ceiling. Redundant.
8. **SwiGLU + GPTQ compression:** 19.6MB artifact, cannot fit 16MB. Fundamental compression gap.
9. **QAT percentile clip mismatch fix:** No measurable gain.
10. **15 GPTQ percentiles (vs 5):** No gain.
11. **train_seq_len=1024:** Catastrophic quality loss from RoPE extrapolation failure.
12. **Bigger batch (1.5x tokens/step):** Fewer steps offset richer gradients. Net negative.
13. **Single GPU Muon training:** Muon requires distributed all-reduce. Grad accum not equivalent.
14. **Gravity (auxiliary loop losses) at low step counts:** Pure noise at 300 steps.
15. **Crawler bank at U-Net bottleneck (H4):** Per-step better, net worse. Steps beat tricks.
16. **Gate on C-steps only:** -0.006 BPB. Not enough training signal.
17. **Detached EMA as PD consensus reference:** Goes stale. One-way gradient kills signal.
18. **temp_scaling (temperature search):** Optimal T=1.000 every time. No effect.
19. **XSA on all 11 layers for submissions:** +0.0006 BPB but +400KB artifact. Over budget.
20. **576plus edge autoresearch:** All 12 runs timed out. Infrastructure problem, no data.

---

## Architecture Decisions (why we chose what we chose)

### Why 11L/512d
- 11 layers is the sweet spot for 600s/8xH100 at ~85ms/step -> ~7000 steps.
- 9 layers undertrained (too few params at 512d). 12 layers: faster per step but worse pre-quant.
- 512d is the largest dim that gives head_dim=32 with 16 heads (FA3 compatible). 480d gives head_dim=30 (invalid).
- U-Net (5 encoder + 6 decoder) with skip connections provides encoder/decoder structure.

### Why LeakyReLU-squared (slope 0.5)
- Tested against standard ReLU-squared. -0.0008 BPB improvement (1.1195 vs 1.1203, seed 1337).
- Leaky variant avoids dead neurons while maintaining the sparsity benefit of squared activation.
- Source: F1 Legal LB results.

### Why XSA last 4 (not all 11)
- XSA-11 gives -0.0006 BPB but makes artifact 400KB larger (16.02MB, over limit).
- XSA-4 provides most of the benefit while staying under 16MB budget.
- The last 4 layers benefit most from extended softmax attention because they're closest to the output.

### Why BigramHash 1536 (not 2048)
- Quality-neutral vs 2048. Saves ~400KB artifact size.
- Enables size headroom for other features (n-gram cache, GPTQ overhead).

### Why ROPE_DIMS=24
- Part of the Podracing SOTA config. ROPE 24 (vs default 16) gives more positional dimensions.
- Used in the verified 0.9625 BPB configuration.

### Why GPTQ (not naive int6)
- Single biggest post-training improvement: -0.0027 BPB.
- Hessian-aware error compensation. Column reordering by ascending Hessian diagonal.
- Block-128, percdamp=0.01, 256 calibration samples from training data.
- 0 naive fallback layers (all 66 layers GPTQ-calibrated).

### Why Muon optimizer (not AdamW for main training)
- Muon with distributed all-reduce is the standard for this competition.
- lr=0.025 (matrices), 0.035 (embeddings), 0.025 (scalars).
- Momentum 0.99, WD 0.04, warmup 1500 steps, warmdown 3500 iters.
- AdamW is only viable for TTT post-training (and even then, SGD is better on relu-squared).

### Why no TTT in current SOTA
- TTT was banned by competition rules (issue #402).
- Even before the ban, legal score-first TTT added at most +0.0003 BPP.
- N-gram eval provides 10x more improvement (0.16 BPB) than TTT ever did.

### Why 7-gram backoff with entropy-adaptive alpha
- Score-first, backward-looking: legal under competition rules.
- Multi-order backoff (orders 2-7): try longest context first, cascade down on miss.
- Entropy-adaptive: trust n-gram more when model is uncertain.
- Formula: `alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))` where H = model entropy.
- This single eval-time technique provides the entire gap from 1.12 to 0.96.
- Credit: n-gram concept @deanbrr (PR #659), backoff + adaptive alpha @Asukabot0 (PR #727).

---

## Competition Rules & Legality Notes

### Constraints
- Artifact size: <=16MB (code + quantized weights + compression)
- Training time: <=10 minutes on 8xH100 SXM
- Metric: bits-per-byte (BPB) on FineWeb validation set
- Challenge window: March 18 - April 30, 2026
- Repo: https://github.com/newjordan/parameter-golf

### Score-First Protocol (CRITICAL)
- **LEGAL:** Score chunk i FIRST, THEN train on chunk i. (The `eval_val_sliding_ttt()` pattern)
- **ILLEGAL:** Train on ALL val data for N epochs, THEN score. (The old `ttt_adapt()` pattern)
- Any TTT that trains on val data before scoring violates issue #402.
- Default to TTT_ENABLED=0 unless score-first sliding window is confirmed in the code.
- The SwiGLU 1.0763 and 1.0756 scores were INVALID (illegal TTT).

### TTT Legality
- TTT is now effectively banned/deprecated for submissions.
- Even legal score-first TTT adds at most +0.0003 BPP.
- All historical TTT results are for research reference only.

### N-gram Eval Legality
- Cache built from already-scored tokens only (backward-looking).
- Alpha depends solely on model's own softmax entropy -- no target/label access.
- No oracle selection, no min-NLL comparison.
- GPTQ calibration runs inside training phase (before wallclock stop).
- Fully compliant with issue #402.

### Submission Checklist (CRITICAL -- PR #674 was CLOSED for missing files)
Every PR must include:
1. `submission.json` (author, github_id, name, blurb, date, val_loss, val_bpb, bytes_total, bytes_code)
2. Training logs for all seeds
3. `README.md` with results table and reproduce instructions
4. `train_gpt.py` in the records folder

File structure: `records/track_10min_16mb/YYYY-MM-DD_Name_Hardware/`

### Multi-Seed Requirements
- SOTA claims require p < 0.01 significance with multiple seeds.
- 3-seed mean is the standard. 2-seed is minimum for preliminary claims.
- Compression is seed-dependent: seeds 7 and 137 busted 16MB on some configs while seeds 1337 and 42 passed.

---

## File Integrity

### SOTA File
- Hash: 147bbccc (96,116 bytes)
- Source: `concepts/podracer/sota/train_gpt.py`

### Verified Copies (NEVER delete)
- `concepts/podracer/sota/` -- current SOTA with run script
- `concepts/podracer/backup1/` -- backup copy
- `concepts/podracer/backup2/` -- backup copy
- `concepts/podracer/backup3/` -- backup copy (train_gpt.py)
- `concepts/podracer/backup4/` -- backup copy (train_gpt.py)
- `concepts/podracer/sota_verified/` -- verified copy
- `records/track_10min_16mb/2026-03-25_PodracingII_backoff7gram_8xH100/train_gpt.py` -- frozen submission copy
- `records/track_10min_16mb/2026-03-25_PodracingII_backoff7gram_8xH100/frozen_sota/train_gpt.py` -- frozen SOTA reference

### GS (Gold Standard) v7
- `GS/GS_train_gpt_v7_1.1206.py` -- GPTQ baseline (1.1206 BPB, PR #508)
- `GS/REPRODUCE.md` -- reproduction instructions

### Key Checkpoints
- `final_model.int6.ptz` -- current quantized model
- `final_model.intq.ptz` -- current int-quant model
- `final_model.pt` -- current float model
- `checkpoints/` -- historical checkpoints directory

---

## Experiment Timeline

| Date | Milestone | BPB | Source |
|------|-----------|-----|--------|
| 2026-03-17 | Naive baseline (9L/512d) | 1.2244 | `records/track_10min_16mb/2026-03-17_NaiveBaseline/` |
| 2026-03-18 | 4-hour unlimited baseline | 1.2074 | `records/track_non_record_16mb/` |
| 2026-03-18 | Fractal experiments (DGX Spark) | 2.5953 | `RESULTS.md` |
| 2026-03-20 | FarnsworthEngine v1 (SOTA254 + TTT) | 1.1303 | `sota254/README.md` |
| 2026-03-21 | Qwen overnight sweep (141 runs) | 2.3332 (local) | `RESULTS.md` |
| 2026-03-21 | SOTA254 improvement experiments | 1.1295 | `records/track_10min_16mb/2026-03-22_SpongeBath_TTT8_Stride32/` |
| 2026-03-22 | Leapfrog campaign (12+ findings) | 1.1232 | `records/leapfrog_results_20260322.md` |
| 2026-03-22 | PR #445 submitted (v1, TTT burst) | 1.1232 | `records/leapfrog_results_20260322.md` |
| 2026-03-22 | Frugendorff v1 (3x4 fractal) | 1.2113 | `RESULTS.md` |
| 2026-03-23 | v7 GPTQ + TTT EMA (PR #508) | 1.1206 | `records/track_10min_16mb/2026-03-23_11L_GPTQ_TTT_EMA_QAT_1.1206/` |
| 2026-03-23 | Frugendorff Squared (6x2) | 1.1478 | `records/track_10min_16mb/2026-03-23_Frugendorff_Squared_6x2_640d_MLP4/` |
| 2026-03-23 | SwiGLU F1 (over budget) | 1.1208 (20.6MB) | `records/track_10min_16mb/2026-03-23_SwiGLU_F1_VRL_LeakyReLU_1.1208/` |
| 2026-03-23 | SwiGLU + AdamW TTT (illegal, over budget) | 1.0763 (19.6MB) | `records/leapfrog_results_20260322.md` |
| 2026-03-24 | F1 Legal LB (3-seed) | 1.1195 | `records/track_10min_16mb/2026-03-24_F1_LegalLB_XSA4_BG1536_1.1195_candidate/` |
| 2026-03-24 | Micro crawler experiments (Runs 1-8) | 1.1325-1.1415 | `MICRO_CRAWLER_RESULTS.md` |
| 2026-03-24 | Cadence ablation (H1+H2) | cad0 wins | `experiments/H1_cadence_characterization/` |
| 2026-03-24 | Crawler bank at U-Net (H4) | per-step better, net worse | `experiments/H4_crawler_bank_on_unet/` |
| 2026-03-24 | World record discovery: n-gram eval | ~1.04 | session state memory |
| 2026-03-25 | **Podracing II (PR #753)** | **0.9625** | `records/track_10min_16mb/2026-03-25_PodracingII_backoff7gram_8xH100/` |

---

## Micro Crawler Full Results (8xH100 SXM, 600s, seed 1337)

Architecture: 4 flat + 2 crawler x 2 = 8 effective depth, dim=640, 10H/5KV, MLP 4x

| Run | Config | Sliding BPB | Post-EMA | Quant Gap | Steps | ms/step | Artifact | Quant |
|-----|--------|-------------|----------|-----------|-------|---------|----------|-------|
| Run 1 | Broken LR, no gate, trigram 8192 | **1.1377** | 1.1513 | 0.0097 | 7,694 | 78 | 16.86MB | per-row |
| Run 1.5 | lr_mul fix + recursive cadence | 1.1384 | 1.1520 | 0.0097 | 7,313 | 82 | 16.33MB | per-row |
| Run 3 | Self-ref gate (C only) + GPTQ | 1.1415 | 1.1575 | 0.0072 | 7,150 | 84 | 16.33MB | GPTQ |
| **Run 6** | **PD gate (EMA) + GPTQ** | **1.1375** | 1.1535 | 0.0075 | 7,076 | 85 | 16.65MB | GPTQ |
| Run 8 | Bidir PD + fixed cad2 + GPTQ | 1.1355 | 1.1522 | 0.0075 | 6,839 | 85 | 17.04MB | GPTQ |
| **cad0** | **No C-steps, GPTQ** | **1.1325** | **1.1487** | **0.0070** | **7,856** | **76** | ~16.5MB | GPTQ |

---

## Cadence Ablation Full Results (0.25 scale, 150s, 8xH100)

### 4f+2cx2 (H1)
| Cadence | Steps | step_avg | val@500 | sliding_bpb | quant_gap |
|---------|-------|----------|---------|-------------|-----------|
| cad1 | 702 | 213ms | 1.3842 | 1.5092 | 0.136 |
| cad2 | 810 | 185ms | 1.3841 | 1.4222 | 0.081 |
| cad3 | 854 | 176ms | 1.3839 | 1.3941 | 0.061 |
| cad4 | 878 | 171ms | 1.3838 | 1.3836 | 0.059 |

### 3f+3cx2 (H2)
| Cadence | Steps | step_avg | val@500 | sliding_bpb | quant_gap |
|---------|-------|----------|---------|-------------|-----------|
| cad1 | 612 | 245ms | 1.3876 | 1.6007 | 0.196 |
| cad2 | 738 | 204ms | 1.3822 | 1.4587 | 0.099 |
| cad3 | 792 | 189ms | 1.3828 | 1.4211 | 0.078 |
| cad4 | 822 | 183ms | 1.3815 | 1.4030 | 0.066 |

### Full Scale Production (600s)
| Config | Steps | step_avg | Memory | sliding_bpb | quant_gap |
|--------|-------|----------|--------|-------------|-----------|
| Run 8 (cad2) | 7,076 | ~85ms | 33.2 GiB | 1.1355 | 0.0075 |
| **cad0 (no C)** | **7,856** | **76ms** | **22.9 GiB** | **1.1325** | **0.0070** |

---

## Competition Landscape (as of 2026-03-25)

| PR | Author | BPB | Key Technique |
|----|--------|-----|---------------|
| #753 (ours) | Frosty40 | **0.9625** | 7-gram backoff + entropy-adaptive alpha |
| #727 | @Asukabot0 | ~0.96 | N-gram backoff (inspiration) |
| #706 (ours) | Frosty40 | ~1.02 | Podracing I (order 5, alpha 0.2) |
| #659 | @deanbrr | ~1.05 | N-gram eval cache concept |
| #587 | ours | 1.1203 | XSA-11 clean |
| #533 | ours | 1.1207 | GPTQ + SGD TTT (XSA-4) |
| #508 | ours | 1.1215 | GPTQ + early QAT + TTT EMA (3-seed) |
| #505 | @JoeProAI | 1.1181 | SwiGLU + NO TTT |
| #503 | @EthanYangTW | 1.1195 | GPTQ + AdamW TTT + XSA-all |
| #473 | @abaybektursun | 1.1214 | Parameter Banking + SGD TTT |
| #445 | ours | 1.1232 | TTT burst + EMA |
| #414 | @signalrush | 1.1233 | Base architecture (11L/512d) |

---

## Infrastructure Notes

- **Hardware:** 8xH100 SXM 80GB HBM3
- **Local dev:** DGX Spark GB10, 130.7GB unified VRAM (no torch.compile, no Triton)
- **Cloud:** RunPod (FA3 + compile working) or Vast.ai (cheaper, H100 ~$1.67/hr)
- **Vast.ai migration:** API key in `~/.vast_api_key`, SSH key `~/.ssh/id_ed25519_apollo`
- **ALWAYS destroy Vast instances after pulling results** (storage charges continue)
- **FA3 requirement:** FlashAttention 3 (Hopper, bf16+hdim64 selective build)
- **H5 Cubric blocked on Vast.ai** (torch.compile + FA incompatibility). Use RunPod instead.
