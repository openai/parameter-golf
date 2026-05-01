# Non-Record Submission: kv4_both Attention-Gate Stack + Mini Recurrence + EMA
*Single-author pipeline, 1×H100 ablation → 8×H100 confirmation, with full attention-component ablation data*

## About me

I'm a software engineer at Cisco. My team works on a enterprise cloud VPN product (remote-access VPN on AWS), mostly Kubernetes and Go. Distributed systems, not ML.

I did deep learning coursework during my master's (graduated December 2021), but hadn't really touched ML since. Parameter Golf seemed like a reasonable way to get back into it — a scoped challenge with a clear metric and a deadline. I worked on this on evenings and weekends over the last month of the challenge, mostly on a single rented H100, with a few 8×H100 runs at the end to confirm the results.

I went in knowing I wouldn't be competitive with the leaderboard frontier. I just wanted to put together something reproducible, learn the stack, and submit one honest non-record entry. Most of this submission is standing on the shoulders of the PRs cited below.

**3-seed mean val_bpb: 1.16133 ± 0.00247** (8×H100, 600s training, stride=64 sliding eval, 14.87 MB mean artifact)

This is a non-record submission. It does not beat SOTA (~0.94 BPB). What it offers the community is:

1. A **clean, reproducible 3-seed result** (std 0.00247) stacking merged techniques on top of the post-04-09 baseline recipe.
2. **Systematic attention-component ablation data** on the `USE_NEW_ATTENTION` stack — isolating which additions actually move BPB.
3. **An engineering finding**: bf16 EMA state silently truncates small-magnitude weights toward zero over thousands of `mul_(0.9965)` steps, hollowing out the model at swap time. Fix: keep EMA state in fp32 regardless of live-param dtype.
4. **Empirical 1-GPU → 8-GPU calibration data** from a single researcher's workflow (as opposed to cloud-credit-rich teams).

---

## Headline Result

```
Config:        kv4_both + EMA
Architecture:  10L × 512d × 8H/4KV (GQA), MLP 3×, SP8192, seq_len 2048
                Parallel residuals from layer 5 (GPT-J style)
                Mini depth recurrence on layers 4,5 (each runs twice)
                Per-head attention additions: q_norm_scale, k_norm_scale,
                    k_gain (on), head_gate (on)
Training:      MuonEq-R-style (Muon + decoupled WD 0.025), MATRIX_LR 0.05,
                EMBED_LR 0.8, WARMDOWN_ITERS 400, EMA decay 0.9965
Eval:          Sliding window, stride=64
Quantization:  Mixed int5 (MLP) / int6 (attn) / int8 (embeddings) + zstd-22
Hardware:      8×H100 SXM, 600s training cap

3-seed post-quant val_bpb:
  seed=42    → 1.1639
  seed=1337  → 1.1590
  seed=2024  → 1.1611
  mean:        1.16133   std: 0.00247
  artifact:    14.87 MB (avg)   train time: 600s   eval time: ~75s
```

Improvement over README baseline (1.2244): **−0.0631 BPB**.
Improvement over same stack without EMA: **−0.00828 BPB** (real, matches leaderboard evidence).

---

## The Stack, with Full Attribution

Nothing in this stack is methodologically novel. Every component has prior art:

| Technique | Credit |
|---|---|
| SP8192 tokenizer | PR #1394 (@clarkkev) |
| Parallel residuals from layer 5 | PR #1204 (@msisovic, merged 2026-04-09), PR #1412 (@Robby955) |
| Mini depth recurrence on layers 4,5 | PR #1204 (@msisovic) — identified layers 4,5 as "U-Net hinge sweet spot" |
| Muon weight decay (decoupled, 0.025) | PR #1445 (@X-Abhishek-X), PR #1421 |
| Sliding-window eval (stride=64) | PR #169, standard in records 2026-03-19 onward |
| Int8 embeddings (from int6) | 04-09 record recipe (SP8192-aware quantization policy) |
| Per-head QK gain (q_gain) | PR #1770, PR #1809, standard in modern stacks |
| Per-head attention output gate (head_gate) | PR #1667 "AttnOutGate", PR #1693, PR #1880, PR #1790, PR #1826 |
| EMA for eval (decay 0.9965) | PR #1445, PR #1421, PR #1471, PR #1406, 04-09 record |
| 10L × MLP 3× shape | PR #331, PR #583, PR #137, PR #286 |
| Overall recipe shape | 04-09 record (records/track_10min_16mb/2026-04-09_…) |

My contribution is the **specific combination + ablation data**, not any individual component.

---

## Ablation Grid (1×H100, 2500 steps, disjoint eval, single seed)

All runs fixed at 2500 training steps for clean comparison. BPB is post-quant (zstd-22 roundtrip). Sorted by BPB.

| Config | USE_NEW | kv | k_gain | head_gate | BPB (post-quant) | Size (MB) | Δ vs baseline_kv4 |
|---|---|---|---|---|---|---|---|
| **kv4_both** | 1 | 4 | ✓ | ✓ | **1.26321** | ~15.0 | **−0.00649** |
| kv4_headgate | 1 | 4 | ✗ | ✓ | 1.26645 | 14.99 | −0.00325 |
| new_attn_kv4 | 1 | 4 | ✗ | ✗ | 1.26976 | 15.13 | −0.00 (flat) |
| **baseline_kv4** | 0 | 4 | — | — | 1.26970 | — | 0 (reference) |
| kv4_kgain | 1 | 4 | ✓ | ✗ | 1.27015 | 15.04 | +0.00045 (flat) |
| kv2_both | 1 | 2 | ✓ | ✓ | 1.26881 | — | −0.00089 |
| kv2_kgain | 1 | 2 | ✓ | ✗ | 1.27507 | 15.94 | +0.00537 |
| new_attn_kv1 | 1 | 1 | ✗ | ✗ | 1.28087 | 14.85 | +0.01117 |
| new_attn_kv2 | 1 | 2 | ✗ | ✗ | 1.28137 | 16.00 | +0.01167 |

### Findings

1. **`head_gate` is the one that matters.** Alone (`kv4_headgate`) it gives −0.003 BPB. Combined with `k_gain` it reaches −0.006 (`kv4_both`). Within single-seed noise at step 2500, but the direction is consistent.

2. **`k_gain` alone is ~flat.** `kv4_kgain` lands at +0.00045 BPB vs baseline — indistinguishable from noise. Per-head K scaling is redundant with per-head Q scaling (`q_gain`) that already exists in the baseline.

3. **`q_norm_scale` / `k_norm_scale` (always-on in `new_attn_kv4`) is neutral.** `new_attn_kv4` lands within 0.00 of baseline. The post-RMSNorm per-head scales are mathematically redundant with the `q_gain`/`k_gain` multipliers downstream.

4. **KV-head reduction hurts at 2500 steps.** kv=4 → kv=2 costs +0.012 BPB; kv=1 (MQA) costs +0.011. Size savings from fewer K/V projections do not offset BPB degradation at this step budget. Interestingly, `kv2_kgain` and `kv2_both` land better than `new_attn_kv2` — implying `head_gate` specifically compensates for part of the KV-reduction cost.

5. **Artifact size is non-monotonic in KV reduction.** `new_attn_kv2` is actually *larger* than `new_attn_kv4` (16.00 vs 15.13 MB). Fewer KV heads = less parameter redundancy = worse zstd compression ratio. Real parameter count is smaller; compressed bytes are not.

---

## Engineering Finding: bf16 EMA Silently Collapses the Model

First EMA implementation stored EMA state in the same dtype as the live params (mix of bf16 and fp32). Post-swap post-quant BPB was **1.3363** — 0.17 BPB worse than pre-swap live BPB of 1.1615.

Artifact size dropped from ~14.9 MB to **13.62 MB** — the giveaway. EMA-weighted weights were compressing *too well*, which only happens if the weights are low-entropy (near-zero).

Root cause: in bf16 (3-decimal precision), `ema_buf.mul_(0.9965)` followed by `add_(p, alpha=0.0035)` silently truncates small-magnitude contributions. Over 7000+ steps, params of magnitude near `1e-3` decay toward zero. The final swap installs an effectively hollowed-out model.

Fix: EMA state is now held in **fp32** regardless of live-param dtype. Per-step update upcasts `p` to fp32 before the weighted add; final swap casts back to the live dtype.

```python
# Before (broken):
ema_state[name] = p.detach().clone()  # inherits p's dtype, often bf16
# ... each step:
ema_buf.mul_(decay).add_(p.detach(), alpha=1-decay)  # bf16 arithmetic → truncation

# After (fixed):
ema_state[name] = p.detach().to(dtype=torch.float32).clone()
# ... each step:
ema_buf.mul_(decay).add_(p.detach().to(dtype=torch.float32), alpha=1-decay)
# ... final swap:
p.data.copy_(ema_buf.to(dtype=p.dtype))
```

Memory cost: ~100 MB extra GPU state (model is ~27.8M fp32 params). Well inside 80 GB H100 budget.

---

## 1-GPU → 8-GPU Calibration

For iteration, I ran ablations on a single H100 at 2500 steps with disjoint eval (`EVAL_STRIDE=0`). Validated each promising candidate on 8×H100 at 600s wallclock with stride=64 eval.

Empirical calibration from matching configs:

| | 1-GPU 2500-step live BPB | 8×H100 600s post-quant BPB | Δ |
|---|---|---|---|
| `kv4_both`, no EMA | 1.2464 | 1.1696 | −0.077 |

Loose rule of thumb: `8×H100 post-quant BPB ≈ 1-GPU 2500-step live BPB − 0.06 to −0.09`. Useful for estimating whether a candidate is worth promoting, but not precise enough for fine deltas.

---

## Result Trajectory

| # | Config | Hardware | Post-quant BPB | Size (MB) |
|---|---|---|---|---|
| 1 | README baseline (9L × 512d × MLP2×, SP1024) | 8×H100 | 1.2244 | — |
| 2 | + 10L × MLP3× × SP8192 + parallel L5 + recur 4,5, no WD | 8×H100 | 1.2742 | 16.65 ❌ |
| 3 | + sliding-512 eval, WD=0.015 | 1×H100 (2500 steps) | 1.2757 | 15.03 |
| 4 | Same config, healthy 8×H100 | 8×H100 | 1.2388 | 15.03 |
| 5 | + int8 embeddings, WD=0.015 | 8×H100 | 1.1717 | 16.10 ❌ |
| 6 | int8 embeddings + WD=0.025 | 8×H100 | 1.1798 | 14.88 |
| 7 | + `kv4_both` attention stack | 8×H100 (3-seed) | 1.16961 | 14.87 |
| 8 | **+ EMA (0.9965)** | **8×H100 (3-seed)** | **1.16133** | **14.87** ✓ |

Each row is a reproducible 8×H100 600s run (or 1-GPU iteration step 3).

---

## Negative Results Worth Noting

- **Parallel residuals + mini recurrence at 2500 steps (1 GPU):** flat vs. baseline within noise. The PR #1204 claim of "biggest single lever" is only visible at 4500+ step training on 8×H100, paired with proper quantization.
- **KV-head reduction** (kv=4 → kv=2 or kv=1) hurts BPB at 2500 steps even with new-attention flags enabled; size doesn't reliably decrease either (compression ratio worsens).
- **`q_norm_scale` / `k_norm_scale`** (always-on per-head post-RMSNorm scales): neutral. Redundant with `q_gain` / `k_gain` below them.
- **MUON_WD=0.05** (tried before settling on 0.025): too aggressive at this LR, BPB +0.05 and artifact shrank to 12.65 MB (weights over-decayed to near-zero-entropy distribution). WD × LR effective-decay rate matters more than absolute WD.
- **Seed 1337 had the worst no-EMA BPB (1.17020)**, became the best with-EMA BPB (1.1590). EMA gain varies 2× across seeds — single-seed results can be deceptive.

---

## Reproduction

Three 8×H100 seeds produce the result. RUN_IDs and log files are included:
- `train_seed42.log` (1.1639, seed 42)
- `train_seed1337.log` (1.1590, seed 1337)
- `train_seed2024.log` (1.1611, seed 2024)

Reproduction commands (one seed, substitute `SEED`):

```bash
# Dataset (first time only)
pip install sentencepiece zstandard brotli
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

# One 8×H100 seed, 600s wallclock:
USE_NEW_ATTENTION=1 NUM_KV_HEADS=4 USE_K_GAIN=1 USE_HEAD_GATE=1 \
SEED=42 \
NUM_LAYERS=10 MLP_MULT=3 VOCAB_SIZE=8192 \
PARALLEL_START_LAYER=5 RECUR_LAYERS=4,5 \
TRAIN_BATCH_TOKENS=524288 VAL_BATCH_SIZE=524288 \
WARMUP_STEPS=20 WARMDOWN_ITERS=400 \
MATRIX_LR=0.05 EMBED_LR=0.8 MUON_WD=0.025 \
ITERATIONS=50000 MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=100 \
EVAL_STRIDE=64 EMA_DECAY=0.9965 \
RUN_ID=kv4_both_EMA_42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All three seeds complete within 600s training + ~75s eval on 8×H100 SXM.

---

## Limitations / Honest Self-Assessment

- **Not a record.** 0.10023 BPB above current leaderboard tip (~1.0611).
- **Nothing methodologically novel.** Every component has prior art (cited above). Contribution is the systematic ablation + engineering finding.
- **Single researcher, single pod.** Ablation done on ~10 hours of 1×H100 + ~30 min of 8×H100 time. No cross-configuration grid. No multi-seed ablation (single seed at 2500 steps for ablation grid; 3 seeds only at final 8×H100).
- **Seed count inconsistent between ablation and final**: ablation uses 1 seed (promoting candidates to 3-seed confirmation); not ideal for comparing small deltas in the ablation data itself.
- **The bf16 EMA bug was a real setback.** A run that produced 1.3363 BPB "looks like" EMA is harmful. Without the fp32 fix, the submission would be worse than no-EMA.

---

## Full Experiment History (43 runs)

Every training run I completed during this challenge, grouped by phase. All numbers are post-quant val_bpb (zstd-22 roundtrip, sliding-window eval where applicable). "live" = in-training disjoint eval at the last logged step.

### Phase 1 — Learning the baseline (1000 steps, 1×H100, seed=1337)

Varied one knob at a time from the stock 9L × MLP2× × SP1024 config to see what moved.

| Run | Config | live | pq_bpb | Size MB |
|---|---|---|---|---|
| `ablation_baseline` | stock config | 1.3844 | 1.4081 | 15.02 |
| `ablation_notie` | untied embeddings | 1.3603 | 1.3702 | 17.28 ❌ |
| `ablation_11L` | 11 layers | 1.3846 | 1.4091 | 16.36 ❌ |
| `ablation_d576` | model_dim 576 | 1.3826 | 1.4100 | 17.71 ❌ |
| `ablation_s4096` | seq_len 4096 | 1.3801 | 1.4047 | 15.04 |

Takeaway: most size-increasing knobs (untied emb, 11L, dim 576) blow the 16 MB cap without meaningful BPB payoff at this step count.

### Phase 2 — Parallel-residuals + mini-recurrence grid (1000 steps, 1×H100, seed=42)

Swept `PARALLEL_START_LAYER` × `RECUR_LAYERS` on the 10L × MLP3× × SP8192 base.

| Run | par_layer | recur_layers | live | pq_bpb | Size MB |
|---|---|---|---|---|---|
| `arch_baseline` | off | — | 1.4604 | 1.4832 | 14.80 |
| `arch_par_l4` | 4 | — | 1.4624 | 1.4850 | 14.70 |
| `arch_par_l5` | 5 | — | 1.4617 | 1.4854 | 14.81 |
| `arch_par_l5_recur_5` | 5 | 5 | 1.4639 | 1.4868 | 14.75 |
| `arch_par_l5_recur_56` | 5 | 5,6 | 1.4627 | 1.4858 | 14.69 |
| `arch_par_l6_recur_45` | 6 | 4,5 | 1.4615 | 1.4839 | 14.85 |
| `arch_par_l6_recur_56` | 6 | 5,6 | 1.4630 | 1.4864 | 14.85 |
| `arch_par_l7_recur_456` | 7 | 4,5,6 | 1.4693 | 1.4962 | 14.42 |
| `arch_par_l4_recur_5` | 4 | 5 | 1.4626 | 1.4858 | 14.93 |
| `arch_recur_5` | off | 5 | 1.4610 | 1.4837 | 14.78 |
| `arch_recur_4_5` | off | 4,5 | 1.4596 | 1.4830 | 14.77 |
| `arch_recur_4_5_6` | off | 4,5,6 | 1.4643 | 1.4889 | 14.54 |
| `arch_recur_5_x2` | off | 5 (×2) | 1.4624 | 1.4843 | 14.83 |

**All 13 configs landed within 0.013 BPB.** At 1000 steps, deltas are inside seed noise. Picked `par_l5 + recur_4,5` ("sweet spot" per PR #1204) for longer runs.

### Phase 3 — 2500-step attention-gate ablation (1×H100, disjoint eval, seed=42)

With `par_l5 + recur_4,5` and `MUON_WD=0.025` fixed, swept `USE_NEW_ATTENTION` flags × KV-head count.

| Run | USE_NEW | kv | k_gain | head_gate | live | pq_bpb | Size MB | Δ vs baseline_kv4 |
|---|---|---|---|---|---|---|---|---|
| `exp_baseline_kv4` | 0 | 4 | — | — | 1.2465 | **1.2697** | 15.38 | 0 (ref) |
| `exp_new_attn_kv4` | 1 | 4 | 0 | 0 | 1.2481 | 1.2698 | 15.13 | +0.0001 (flat) |
| `exp_kv4_kgain` | 1 | 4 | 1 | 0 | 1.2492 | 1.2701 | 15.04 | +0.0004 |
| `exp_kv4_headgate` | 1 | 4 | 0 | 1 | 1.2478 | **1.2664** | 14.99 | **−0.0033** |
| `exp_kv4_both` | 1 | 4 | 1 | 1 | 1.2464 | **1.2632** | 15.03 | **−0.0065** |
| `exp_kv2_both` | 1 | 2 | 1 | 1 | 1.2522 | 1.2688 | 16.27 ❌ | −0.0009 |
| `exp_kv2_kgain` | 1 | 2 | 1 | 0 | 1.2526 | 1.2751 | 15.94 | +0.0054 |
| `exp_new_attn_kv2` | 1 | 2 | 0 | 0 | 1.2541 | 1.2814 | 16.00 ❌ | +0.0117 |
| `exp_new_attn_kv1` | 1 | 1 | 0 | 0 | 1.2573 | 1.2809 | 14.85 | +0.0112 |

**Findings:**
- `head_gate` alone: **−0.0033 BPB** — weak but consistent direction.
- `k_gain` alone: flat (+0.0004). Redundant with `q_gain`.
- Both together (`kv4_both`): **−0.0065 BPB**. At edge of single-seed noise.
- `q_norm_scale` + `k_norm_scale` (always-on when USE_NEW_ATTENTION=1): neutral. Redundant with the `q_gain`/`k_gain` multipliers below them in the forward pass.
- KV-head reduction hurts (+0.011 to +0.012 BPB). Artifact size non-monotonic with KV count (lower parameter redundancy compresses worse).

Picked `kv4_both` for 8×H100 promotion.

### Phase 4 — Early 8×H100 runs (before new-attention stack)

| Run | WD | live | pq_bpb | Size MB | Notes |
|---|---|---|---|---|---|
| `multi_par_l5_recur_45` | 0.015 | 1.1589 | 1.2388 | 15.03 | baseline training knobs + sliding-64 eval |
| sliding_eval (int8 embed) | 0.015 | 1.1604 | 1.1717 | **16.10 ❌** | int8 embeddings added → over cap |
| sliding_eval (int8, WD 0.025) | 0.025 | 1.1654 | **1.1798** | **14.88 ✓** | int8 + tighter WD → under cap |

First submittable result: **1.1798 post-quant, 14.88 MB.** Int8 embeddings were the single biggest lever (~−0.07 BPB), paired with WD 0.025 to stay under the cap.

### Phase 5 — 3-seed 8×H100 kv4_both (no EMA)

| Run | seed | live | pq_bpb | Size MB |
|---|---|---|---|---|
| `exp_kv4_both_8xh100` | 42 | 1.1734 | **1.1696** | 14.88 |
| `exp_kv4_both_8xh100_1337` | 1337 | 1.1738 | **1.1702** | 14.95 |
| `exp_kv4_both_8xh100_2024` | 2024 | 1.1730 | **1.1690** | 14.86 |

**3-seed mean: 1.1696, std 0.00061.** Tight cluster.

### Phase 6 — 3-seed 8×H100 kv4_both + EMA (final)

After fixing the bf16 EMA truncation bug (see Engineering Finding above):

| Run | seed | live | pq_bpb | Size MB |
|---|---|---|---|---|
| `exp_kv4_both_8xh100_EMA_42` | 42 | 1.1642 | **1.1639** | 14.88 |
| `exp_kv4_both_8xh100_EMA_1337` | 1337 | 1.1595 | **1.1590** | 14.87 |
| `exp_kv4_both_8xh100_EMA_2024` | 2024 | 1.1613 | **1.1611** | 14.86 |

**3-seed mean: 1.16133, std 0.00247.** ← Headline number.

EMA's contribution on the mean: **−0.00828 BPB** (1.16961 → 1.16133). Per-seed EMA gain varied nearly 2× (−0.0058 to −0.0112); seed=1337 went from worst no-EMA to best with-EMA.

### Totals

- **43 completed training runs** across 6 phases
  - 13 × 1000-step Phase-2 arch grid (1×H100, seed=42)
  - 5 × 1000-step Phase-1 shape ablations (1×H100, seed=1337)
  - 9 × 2500-step attention ablation (1×H100, seed=42)
  - 3 × 8×H100 Phase-4 baseline-stack runs
  - 3 × 8×H100 no-EMA kv4_both (3 seeds)
  - 3 × 8×H100 EMA kv4_both (3 seeds) — **final**
  - 7 × smoke/pipeline verification runs
- Plus a handful of aborted UUID-tagged runs during early pod setup (not counted)

---

## Included Files

- `README.md` (this file)
- `train_gpt.py` (the training script with EMA + kv4_both stack — copied from `train_gpt_exp.py`)
- `submission.json` (3-seed result metadata)
- `train_seed42.log`, `train_seed1337.log`, `train_seed2024.log`
- `ablation_results.json` (full 9-config 1-GPU ablation data)
- `experiment_history.tsv` (all 43 runs with config + BPB + size)

---

## Thanks

To the authors of the PRs I built on: @msisovic (#1204), @clarkkev (#1394), @X-Abhishek-X (#1445, #1471), @Robby955 (#1412), and the author of #1667 for AttnOutGate. The 04-09 merged record (records/track_10min_16mb/2026-04-09_…) provided the foundational recipe shape that all post-Apr-09 submissions build on.
