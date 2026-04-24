# Experiment Journal

Only what can be verified from saved logs. Below I point to files in [../records/](../records/) of the fork and to files from my local workspace. Where a number comes from my notes rather than a training log, I mark it plainly.

## What I actually ran

Four real H100 runs plus one local baseline on a 3090. That's the complete list of what I can prove with logs. Intermediate configurations exist as scripts and sketches, but without saved training logs I don't count their numbers as mine.

## Local baseline on 3090 Ti

File: `experiments/001_baseline/train.log` (in my local workspace, not in this repo).

Ran on two RTX 3090 Ti (world_size=2, grad_accum=4) just to verify the upstream baseline builds and trains. Standard configuration: 9 layers, dim=512, MLP hidden=1024, vocab=1024.

Results:
- step_avg: 1208 ms (for comparison, 8×H100 does 54 to 60 ms)
- step 4000, val_bpb: **1.2954**
- Run stopped after 4200 steps, not taken to the end

Takeaway: on my local hardware you can't hit a competitive result in 600 seconds. Baseline served as a sanity check. For submissions I rented 8×H100 SXM on RunPod.

## 020_ultimate. A documented failure

Files: `experiments/020_ultimate/` (workspace), including `README.md`, `eval_results.json`, `train_*.log` (one final run plus seven calibration runs).

The idea: pack into one script all the techniques from upstream PRs I'd read to that point. Twelve layers of SwiGLU(1024), Exclusive Self-Attention on layers 8 to 11, Chunked Window Attention with warmup (64 to 1024), Partial RoPE (50%, base=50000), EMA (decay=0.999, last 25%), Label Smoothing 0.05, BigramHash 8192×96, Spectral init for embeddings, Mixed INT5/INT6 STE quantization.

Eleven new components in one script. Anyone who's written code for more than two years would tell you this is a bad idea.

Calibration runs (50 steps each, picking the sizes):
- dim=576: 2.9 bpb, artifact 6.4 MB PASS
- dim=768: 2.9 bpb, artifact 9.2 MB PASS
- dim=1024 + hidden=2880: 2.9 bpb, artifact 13.4 MB PASS
- dim=1024 + hidden=3520: 2.88 bpb, artifact 12.8 MB PASS
- dim=1024 at 2000 steps: pre-quant 1.57, post 1.67, artifact 38 MB **FAIL:OVER_16MB**

Final run (train_compile_fix.log):
- 6 601 steps in 622 seconds, step_avg 86 ms
- Pre-quant val_bpb: 1.57
- Post-quant val_bpb: **1.4143**
- Artifact: 13.17 MB PASS

Post-mortem from the `README.md` of that experiment plus rereading the logs:

1. SwiGLU is 3 matmuls in the MLP instead of 2. On 12 layers that's a 37% hit on step count at fixed wall time. 6 601 instead of roughly 11 000 with my second submission.
2. Window attention with warmup 64 to 1024 doesn't converge in 600 seconds.
3. EMA with decay=0.999 is worse than SWA for quantization: the last weights with tiny LR dominate the average, giving a sharp optimum that quantization breaks.
4. Label Smoothing under aggressive quantization blurs logits on weak classes.
5. Too many untested features at once. When the model outputs pre-quant 1.57, I can't tell which of the 11 features is guilty.

Lesson: iterative method isn't a luxury, it's the only method.

## 025_optimized → PR #370. First submission

Files: [records/track_10min_16mb/2026-03-21_MixedQuant_BigramHash_SWA/](../records/track_10min_16mb/2026-03-21_MixedQuant_BigramHash_SWA/) in the fork, full copy in `experiments/025_optimized/` in workspace.

Put this configuration together after the `020_ultimate` failure. Principle: take only the proven features from my failed script, drop everything else. Removed SwiGLU (back to ReLU² 3x), XSA, Chunked Window Attention, Partial RoPE, Label Smoothing. Kept: 11 layers, BigramHash(10240), SmearGate, Orthogonal init, Muon WD=0.04, SWA every 50 steps from 50% of training, Mixed INT6/INT8 STE, zstd-22.

Parameters:
- 11 layers × 512 dim × 8 heads × 4 KV heads (GQA)
- MLP ReLU² 3x expansion (hidden=1536)
- vocab_size=1024, train_seq_len=1024
- matrix_lr=0.02, scalar_lr=0.04, tied_embed_lr=0.05
- Muon momentum warmup 0.85 to 0.99 over 1500 steps
- BigramHash 10240 buckets × 128 dim
- Mixed quant: int6 on weights, int8 on embeddings, STE

Run (train.log):
- 11 070 steps in 600 seconds, step_avg 54.2 ms
- val_bpb trajectory (every 1000 steps): 1.399 to 1.334 to 1.301 to 1.285 to 1.279 to 1.271 to 1.268 to 1.263 to 1.244 to 1.220 to 1.193 to **1.1924** (pre-quant)
- SWA started: step 5335 (about 48% of total), 115 snapshots averaged
- Post-roundtrip: **1.2421**
- Artifact: 13 279 428 bytes PASS

Quantization gap: 0.0497 bpb. For context: the leader that day held a gap below 0.02. That means my pre-quant is competitive, but STE + mixed precision didn't squeeze the gap down.

Opened [PR #370](https://github.com/openai/parameter-golf/pull/370) in upstream on 2026-03-21 at 20:43 UTC.

## April submission → PR #1205

Files: [records/track_10min_16mb/2026-04-01_TurboMuon_EngramLite_Improved/](../records/track_10min_16mb/2026-04-01_TurboMuon_EngramLite_Improved/)

Changed strategy after 10 days of thinking. Instead of building my own stack, take a well-documented top base and tune hyperparameters with 3-seed verification. Base: [PR #1089](https://github.com/openai/parameter-golf/pull/1089) by @Bortlesboat, Turbo-Muon + EngramLite stack.

My deltas to PR #1089 (seven hyperparameters):

| Parameter | PR #1089 | Mine | Reason |
|---|---|---|---|
| matrix_lr, scalar_lr | 0.025 | 0.030 | Faster convergence in 600 s |
| warmdown_iters | 3500 | 4500 | Smoother weight averaging |
| muon_momentum_warmup_steps | 1500 | 1000 | Reach target 0.99 sooner |
| VE_LAYERS | 9, 10 | 8, 9, 10 | Extra token identity signal |
| NGRAM_BUCKETS | 8192 | 10240 | Wider n-gram coverage |
| NGRAM_DIM_PER_HEAD | 32 | 48 | Denser embedding |

Three seed runs:

| Seed | Steps | Step_avg | Pre-quant | Sliding | Roundtrip | Artifact |
|---|---|---|---|---|---|---|
| 1337 | 5 538 | 106.74 ms | ~1.93 | **1.1425** | 1.1657 | 15 988 293 |
| 42 | 5 572 | 106.09 ms | ~1.93 | **1.1438** | 1.1669 | 15 978 184 |
| 2024 | 5 576 | 106.00 ms | ~1.93 | **1.1431** | 1.1652 | 15 985 158 |
| **Mean** | 5 562 | 106.28 ms | | **1.1431** | 1.1659 | |

Standard deviation on sliding: 0.0007. Quantization gap: 0.023 (substantial improvement vs the March 0.0497).

Training structure from the logs:
- Start: train_loss roughly 2.37 at step 500
- SWA start: step ~4650 across all three seeds
- Late QAT enable: step ~4856 across all three seeds
- Early stop by wallclock_cap at 591 seconds
- 18 SWA checkpoints
- int5 on all 66 weight groups, 20.5% selective pruning, Brotli-11 + byte-shuffle

Opened [PR #1205](https://github.com/openai/parameter-golf/pull/1205) in upstream on 2026-04-01 at 03:18 UTC.

## Four attempts that never made it to submission (v1 through v5)

Between the March and April submissions I spent ten days trying to build a Frankenstein from top PRs (after realizing my `020_ultimate` went that way and failed). Four variants in workspace:

- `v1_safe_merger`: attempt to combine XSA from PR #609, VRL from PR #569, EMA+Partial RoPE from PR #414
- `v2_soft_quant`: soft quantization schedule instead of hard STE
- `v3_vrl_first`: VRL (variance reducing loss) as the main technique
- `v5_ttt_killer`: Test-Time Training via LoRA at eval time

None of them made it to an H100 run with a saved log. Each folder has only `train_gpt.py` (56 to 62 KB) and an empty `logs/`. Reasons by variant:

- v1: pre-calibration math showed the configuration wouldn't fit 16 MB after quantization. Never launched.
- v2: soft quant schedule diverged on a local smoke test after 2000 steps. Didn't go to H100.
- v3: VRL needs a separate calibration I didn't set up.
- v5: TTT via LoRA at eval time has its own problems (the TTT time budget doesn't fit into the 600 s contest rule).

These variants aren't experiments in my book. They're code-reading exercises that never reached a measurable result.

Variant v4 doesn't exist. Superstition.

## What I studied from other participants

My work leans on publicly available upstream PRs. Some ideas I applied directly in my submissions, some I just studied. It matters to separate what's mine from what's not.

Used directly as others' techniques in my submissions:
- Sliding window evaluation stride=64 ([PR #50](https://github.com/openai/parameter-golf/pull/50))
- SmearGate + BigramHash + Orthogonal Init base recipe ([PR #198](https://github.com/openai/parameter-golf/pull/198), [PR #162](https://github.com/openai/parameter-golf/pull/162) by Raahil Shah)
- Mixed INT6/INT8 STE, 10L MLP 3x recipe ([PR #180](https://github.com/openai/parameter-golf/pull/180) by Tianhao Wu)
- Turbo-Muon + EngramLite full stack, base of my April submission ([PR #1089](https://github.com/openai/parameter-golf/pull/1089) by @Bortlesboat)
- GPTQ with Hessian approach ([PR #634](https://github.com/openai/parameter-golf/pull/634))
- LeakyReLU² activation for MLP ([PR #493](https://github.com/openai/parameter-golf/pull/493), [PR #518](https://github.com/openai/parameter-golf/pull/518))

Read and studied but didn't use:
- XSA all-layers recipe ([PR #265](https://github.com/openai/parameter-golf/pull/265), [PR #287](https://github.com/openai/parameter-golf/pull/287))
- GDN Hybrid architecture ([PR #1564](https://github.com/openai/parameter-golf/pull/1564)). The script sits in parameter-golf-scripts
- 3x3090 port of Turbo-Muon ([PR #1477](https://github.com/openai/parameter-golf/pull/1477))
- Various quantization variants from a dozen other PRs

My role in the first submission: composing proven techniques into a new configuration, cutting the `020_ultimate` stack, running 600 s and debugging down to 1.2421.

My role in the second submission: seven tuned hyperparameters on top of PR #1089, 3-seed verification for statistical confidence, an open PR with metadata and seed logs.

## Summary table of verified results

Only runs verified by logs.

| Run | Date | Hardware | Steps | val_bpb final | Artifact | File |
|---|---|---|---|---|---|---|
| 001_baseline (local) | 03-20 | 2×3090 Ti | 4 200+ | 1.2954 at step 4000 | n/a | workspace/experiments/001_baseline/train.log |
| 020_ultimate (failure) | 03-21 | 8×H100 | 6 601 | **1.4143** | 13.17 MB | workspace/experiments/020_ultimate/ |
| 025_optimized (PR #370) | 03-21 | 8×H100 | 11 070 | **1.2421** | 13.28 MB | records/.../2026-03-21_MixedQuant_BigramHash_SWA/train.log |
| Turbo-Muon seed 1337 | 04-01 | 8×H100 | 5 538 | 1.1425 | 15.99 MB | records/.../2026-04-01_.../train_seed1337.log |
| Turbo-Muon seed 42 | 04-01 | 8×H100 | 5 572 | 1.1438 | 15.98 MB | records/.../2026-04-01_.../train_seed42.log |
| Turbo-Muon seed 2024 | 04-01 | 8×H100 | 5 576 | 1.1431 | 15.99 MB | records/.../2026-04-01_.../train_seed2024.log |
| Turbo-Muon mean (PR #1205) | 04-01 | 8×H100 | 5 562 | **1.1431** (σ=0.0007) | ~15.99 MB | mean of 3 seeds |

Everything else is either other people's submissions I studied (honestly marked in the previous section) or local drafts without a training log (v1 to v5).

## Lessons

Four lessons that actually transferred.

1. **Iterative method isn't a luxury.** Add one feature at a time, measure, move on. Otherwise you can't tell what broke.

2. **Pre-quant and post-quant are two different metrics.** Pre-quant shows model potential. Post-quant shows what you'll deliver. The spread between them is your quantization gap. Shrinking it is the most important lever in Parameter Golf after architecture.

3. **SWA beats EMA for quantization.** The reason is geometric. SWA averages snapshots from different stages, giving a flat weight-space optimum. Flat optima are robust to noise, and quantization is noise. EMA biases weights toward the late steps with small LR, giving a sharp optimum.

4. **Don't try to jump over the top in one night before the deadline.** Tune a well-documented top base on 7 hyperparameters and do 3-seed verification, rather than inventing your own stack and hoping.
