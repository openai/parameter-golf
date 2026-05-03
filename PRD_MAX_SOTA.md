# PRD: Parameter Golf Maximum SOTA Attempt

**Author:** Tanish Gudise
**Status:** Active
**Deadline:** April 30, 2026 (6 days from start)
**Current frontier:** PR #1797 by dexhunter, val_bpb 1.06157 (3-seed mean, std 0.00066)
**Current personal best:** 1.07060 BPB on Kevin Clark's #1394 SP8192 base (3-seed, 1×H100)
**Win bar:** 1.0566 BPB or lower, 3-seed mean, p<0.01 vs PR #1797

## Executive summary

Replicate the entire #1797 stack as a foundation, then layer four orthogonal mechanisms on top, each chosen to maximize cumulative gain rather than to "stack safely." Total estimated cumulative gain: -0.005 to -0.018 BPB beyond #1797, depending on lever interactions. Validate on 8×H100 SXM evaluation hardware, statistically validate against #1797's published 3-seed distribution, submit as a record PR.

Probability of the full stack landing at or below 1.0566 (the record bar): honest estimate 8-15%. Probability of landing strictly below #1797 (1.06157, regardless of significance bar): 25-35%. Probability of landing as a credible non-record submission with novel methodology: >85%.

The plan trades safety for ceiling. Safer plans with one or two levers cap at ~1.060. This plan caps at ~1.045 if all four levers stack additively (they probably won't, but ceiling is what we optimize for).

## Goals

**Primary (P0):** Three-seed mean val_bpb ≤ 1.0566 on 8×H100 SXM evaluation, p<0.01 vs #1797's distribution, all artifact/time/eval caps satisfied.

**Secondary (P1):** Even if primary fails, ship a non-record submission with the strongest single-novel-lever result available (per-layer QK-Gain schedule on best base reachable), written up with rigorous methodology.

**Tertiary (P2):** Document every ablation and negative result in a final research note. The competition's hiring component values rigor. Negative results documented well are still resume artifacts.

## Non-goals

- Beating tokenizer-cheating submissions (Scylla #1184, etc.) — those are likely to be disqualified and shouldn't define the bar.
- Beating pre-eval TTT submissions (PROTEUS #568 at 0.7853) — explicitly ruled non-spirit by organizers.
- Beating Multi-Pass Streaming TTT (#573 at 1.0523) — legality disputed, may be ruled illegal.
- Optimizing for the displayed README leaderboard (1.0810) — that lags the PR queue by 2+ weeks.

## Architecture: the stack

The stack has five layers. Each layer has a specific role and a specific failure mode.

**Layer 0 — Base:** PR #1797's full stack as inherited from dexhunter's branch. This is non-negotiable. The frontier is here. Building on Kevin's #1394 alone caps you at ~1.060 even with every lever in this PRD.

**Layer 1 — QK-Gain schedule:** the per-layer attention logit scaling lever validated on 1×H100 grad_accum=8 at 1.07060. Schedule: `2.0,2.5,3.0,3.5,4.0,4.5,4.5,4.0,3.5,3.0,2.5`. Mechanism: scales attention logits per layer to balance early-layer copy behavior vs middle-layer composition vs late-layer prediction. Operates pre-softmax. Independent from gates (which operate post-softmax).

**Layer 2 — OptRot pre-quantization rotation** (arxiv 2512.24124). Rotates weight matrices via learned/computed rotation matrices that redistribute outliers before GPTQ. Fuses into adjacent layers — zero artifact cost. Reduces int6 quantization gap by 30-50% in the paper. Mechanism: orthogonal to all training-time levers, operates only at quantization. Risk: minimal — the rotation is mathematically lossless if done correctly.

**Layer 3 — AdamHD Huber-decay** (arxiv 2511.14721). Replaces Muon's L2 weight decay with Huber regularizer. Quadratic below threshold, linear above. Specifically suppresses outlier weights that dominate int6 quantization error. Independent from QK-Gain (different optimizer behavior) and from OptRot (regularization at training time vs rotation at quantization time). Risk: tuning the threshold matters; wrong threshold can hurt vs vanilla decoupled WD.

**Layer 4 — LaCT (Large Chunk TTT)** (arxiv 2505.23884, ICLR 2026 Oral). Replaces #1797's per-token PhasedTTT with document-sized chunks as the update unit. Paper: 70% GPU utilization vs <5% for per-token TTT. Uses Muon as the fast-weight optimizer (which the competition already uses for training). Mechanism: enables 2-3× more TTT epochs within the eval budget, where each epoch matters because PhasedTTT itself is responsible for ~0.013 BPB of the gain in the existing #1797 stack.

**Why these four:** mechanism diversity. Each operates on a different axis:
- Layer 1: training-time, attention logits, pre-softmax
- Layer 2: post-training, weight rotation, pre-GPTQ
- Layer 3: training-time, optimizer regularization, weight magnitude distribution
- Layer 4: eval-time, TTT update unit and frequency

Diminishing returns happen when two levers extract the same signal. These four extract different signals from different stages.

**Why not more:** more isn't better past 4. Each additional lever adds 1-2 days of integration risk, and the ceiling on independent gain is bounded by the actual quality gap remaining. Past 4 we get into overfitting-to-this-validation-set territory.

## Estimated cumulative gain

Independent estimates from papers and competition data:

| Layer | Lower est. | Upper est. | Confidence |
|---|---|---|---|
| 1: QK-Gain schedule | -0.001 | -0.003 | High (validated locally on #1394 base) |
| 2: OptRot | -0.002 | -0.005 | Medium (paper-only, untested in this competition) |
| 3: AdamHD | -0.002 | -0.005 | Medium (paper-only, untested in this competition) |
| 4: LaCT | -0.003 | -0.010 | Medium-low (paper claims big, integration with PhasedTTT non-trivial) |

**If purely additive (best case):** -0.008 to -0.023 BPB → land at 1.039 to 1.054. Wins comfortably.

**If 50% additive (realistic case):** -0.004 to -0.012 BPB → land at 1.050 to 1.058. Wins narrowly to wins by 0.005-0.011.

**If only 2 of 4 deliver (pessimistic case):** -0.002 to -0.005 BPB → land at 1.057 to 1.060. Doesn't clear the 0.005 record bar but lands near or just above #1797 — still a strong non-record submission.

**If layers interfere (failure case):** stack performs worse than #1797. Submit non-record on simpler base.

## Tasks

Tasks are ordered by dependency, not timeline. Each task has a clear success criterion and a clear abort condition.

### Task 1: Restore working environment

**Owner:** human (user)
**Output:** Pod running, repo current, FineWeb data present, brotli installed, all backups confirmed.

**Steps:**
- Start RunPod pod 8m8vahlsggh04d
- SSH in, `cd /workspace/parameter-golf`
- `git pull origin sp8192-rebase`
- `pip install brotli`
- Verify `final_model_seed42_BACKUP.pt`, `final_model_seed2025_BACKUP.pt`, `final_model_1.0707_BACKUP.pt` all present
- Verify `data/datasets/fineweb10B_sp8192/` populated

**Success:** all checks pass, pod responsive.

### Task 2: Pull and verify PR #1797

**Owner:** Claude Code
**Output:** A working checkout of dexhunter's #1797 branch, smoke-tested on 1×H100.

**Steps:**
- `git remote add upstream https://github.com/openai/parameter-golf.git`
- `git fetch upstream pull/1797/head:dexhunter-1797`
- `git checkout dexhunter-1797`
- Read `records/track_10min_16mb/2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT/README.md` end to end
- Read every `.py` file in their submission folder
- Run `prepare_caseops_data.py` to rebuild CaseOps val shards (these are different from your existing FineWeb shards)
- Verify BOS_ID=1 fix is present in the prep script (commit 04d35ed and similar)
- Smoke test: 100 iterations on 1×H100 with their default seed=314, verify it trains, validates, quantizes, evals end-to-end without crashes
- Capture pre-TTT BPB, post-TTT BPB, artifact size

**Success:** smoke completes; numbers from smoke are roughly proportional to their reported full-run numbers (i.e., not catastrophically worse).

**Abort:** smoke crashes with errors that aren't environmental (brotli, fa3) and can't be fixed in 1 hour. Document the crash, fall back to a less aggressive base (PR #1787 or #1736).

### Task 3: Reproduce PR #1797 fully on 1×H100

**Owner:** Claude Code, monitored by human
**Output:** A 3-seed run of dexhunter's stack on 1×H100 grad_accum=8 with no modifications, establishing your reference numbers.

**Steps:**
- Run their full stack with their seed 314, then 42, then 1234 — exact reproduction
- Each run: ~5 hours train + 11 min eval
- Capture every metric: train_loss, val_loss, val_bpb pre-quant, val_bpb post-quant non-sliding, val_bpb post-quant sliding, val_bpb post-TTT, artifact_size, train_time, eval_time
- Compute mean and std across the 3 seeds

**Success:** 3-seed mean within 0.003 BPB of dexhunter's reported 1.06157 on 1×H100. (Note: 1×H100 will likely come in slightly worse than 8×H100 due to grad_accum vs DDP differences. Acceptable range: 1.062 to 1.066 on 1×H100.)

**Abort:** mean differs from theirs by >0.005 BPB. Indicates your build isn't a faithful reproduction. Stop and debug before adding any levers.

**Why this matters:** without a known-good reference, you can't measure whether your additions help or hurt. Every subsequent comparison is relative to this number.

### Task 4: Implement Layer 1 (QK-Gain schedule)

**Owner:** Claude Code
**Output:** PR #1797 + per-layer QK-Gain schedule on attention.

**Steps:**
- Open the readable train_gpt_human.py for #1797 (or decompress from the wrapper)
- Find `CausalSelfAttention.__init__` and the attention forward pass
- Locate the existing QK-Gain mechanism if present (#1797 inherits from #1493 which uses qk_gain=5.25 globally) or add it if not
- Replace the global qk_gain with a per-layer schedule via env var `QK_GAIN_INIT_SCHEDULE`, parsed as 11 floats
- Default schedule: `2.0,2.5,3.0,3.5,4.0,4.5,4.5,4.0,3.5,3.0,2.5`
- Recompile via the LZMA+b85 wrapper
- Verify code size still under 16,000,000 byte cap
- Smoke test: 100 iterations, verify model initializes with per-layer values and trains without NaN

**Success:** smoke trains cleanly; model dump shows the per-layer values are correctly applied.

**Abort:** model NaNs or schedule values aren't being applied. Debug threading of args through Block → Attention. Common bug: positional vs keyword args in minified source.

### Task 5: Single-seed run with Layer 1 added

**Owner:** Claude Code
**Output:** 1-seed run of #1797 + QK-Gain on 1×H100.

**Steps:**
- Run seed 42 with the modified script
- Compare every metric vs Task 3 reference seed 42

**Success:** post-TTT val_bpb is at least -0.0005 better than reference seed 42. (-0.0005 is the noise floor; real signal needs to be visibly larger.)

**Decision:**
- **Strong signal (-0.001 or better):** continue to Layer 2. QK-Gain stacks on this base.
- **Marginal signal (-0.0001 to -0.001):** continue to Layer 2 but flag as low-confidence layer.
- **Neutral or worse:** Layer 1 doesn't stack on this base. Drop it. Continue to Layer 2 without it.

### Task 6: Implement Layer 2 (OptRot pre-quantization)

**Owner:** Claude Code
**Output:** OptRot rotation applied before GPTQ in the quantization pipeline.

**Steps:**
- Read the OptRot paper (arxiv 2512.24124) carefully — understand rotation matrix construction
- Implement rotation as a pre-GPTQ pass:
  - For each linear layer that gets int6 quantized, compute optimal rotation R
  - Rotate weights: W' = W @ R
  - Fuse R⁻¹ into the adjacent layer (the one consuming this layer's output)
  - Verify mathematical equivalence: forward pass with (W', R⁻¹) ≡ forward pass with W
- Run GPTQ on rotated weights
- Verify int6 reconstruction error decreases vs un-rotated baseline

**Success:** rotated GPTQ reconstruction error is measurably lower (paper claims 30-50% reduction; accept anything ≥10%).

**Abort:** can't get rotation+fuse to be mathematically equivalent. Drop OptRot. The layer is high-value but high-implementation-risk.

### Task 7: Single-seed run with Layers 1+2

**Owner:** Claude Code
**Output:** 1-seed run of #1797 + QK-Gain + OptRot.

**Steps:** as Task 5 with Layer 2 added.

**Success criteria same as Task 5.** Cumulative gain target: -0.003 BPB or better vs Task 3 reference.

### Task 8: Implement Layer 3 (AdamHD)

**Owner:** Claude Code
**Output:** Muon optimizer with Huber decay replacing L2 decay.

**Steps:**
- Read AdamHD paper (arxiv 2511.14721)
- Identify Muon's WD application point in the optimizer code
- Replace L2 decay term with Huber: quadratic below threshold δ, linear above
- Default δ: paper recommends per-parameter tuning; start with δ = 0.1 × mean(|w|) per parameter group
- Verify gradients flow correctly with Huber

**Success:** training is stable, loss curve looks normal in first 100 iterations.

### Task 9: Single-seed run with Layers 1+2+3

**Owner:** Claude Code
**Output:** 1-seed run of #1797 + QK-Gain + OptRot + AdamHD.

**Steps:** as Task 5/7 with Layer 3 added.

**Success criteria same as Task 5.** Cumulative gain target: -0.005 BPB or better vs Task 3 reference. **This is the threshold where the win attempt becomes credible.**

**Decision:**
- **Cumulative ≤-0.005:** continue to Layer 4
- **Cumulative -0.003 to -0.005:** layer 4 is needed but optional — assess whether LaCT integration risk is worth the marginal gain. If yes, continue. If no, skip Layer 4 and go to Task 11.
- **Cumulative >-0.003:** the levers aren't compounding well. Submit what you have as non-record. Skip Layer 4.

### Task 10: Implement Layer 4 (LaCT)

**Owner:** Claude Code
**Output:** PhasedTTT replaced with document-chunk LaCT.

**Steps:**
- Read LaCT paper (arxiv 2505.23884)
- Find PhasedTTT implementation in #1797
- Replace per-token TTT update with document-sized chunk update
- Use Muon as fast-weight optimizer (already available)
- Tune chunk size (paper uses 2K-1M tokens; for FineWeb start with full document chunks)
- Tune epoch count to maximize within 600s eval budget

**Success:** TTT eval completes in ≤600s; post-TTT BPB lower than PhasedTTT reference.

**Abort:** LaCT integration breaks PhasedTTT eval pipeline (BOS handling, byte sidecar, etc.) and can't be fixed in 1 hour. Drop LaCT, finalize stack with Layers 1+2+3.

### Task 11: Single-seed run with full stack (Layers 1+2+3+4 or whatever survived)

**Owner:** Claude Code
**Output:** 1-seed run of full stack.

**Steps:** as Task 5/7/9 with Layer 4 added.

**Success criteria:** cumulative gain ≤-0.005 BPB on 1×H100 vs Task 3 reference.

### Task 12: 8×H100 single-seed validation

**Owner:** Claude Code, monitored by human (because 8×H100 is expensive)
**Output:** 1-seed run of full stack on actual evaluation hardware.

**Steps:**
- Spin up 8×H100 SXM pod (community cloud preferred to halve cost; secure on-demand acceptable if community unavailable)
- Pull current branch
- Install brotli on fresh pod
- Download CaseOps data shards
- Run seed 42 with full stack, grad_accum=1 (since we have 8 GPUs now)
- Capture val_bpb pre-TTT, post-TTT, train time, eval time, artifact size

**Success criteria:**
- val_bpb post-TTT matches the 1×H100 result within 0.003 BPB
- train_time ≤ 600s
- eval_time ≤ 600s
- artifact ≤ 16,000,000 bytes
- val_bpb post-TTT ≤ 1.0566

**Decision:**
- **All criteria met:** proceed to 3-seed validation (Task 13)
- **8×H100 vs 1×H100 numbers don't match:** debug. Likely culprits: gradient accumulation reduction order, FA3 kernel selection, EMA accumulation precision, batch shape changes. Each debug iteration is $10. Allow 2 iterations.
- **train_time >600s:** drop a lever. Most likely culprit is LaCT (TTT eval can spill into eval budget but training shouldn't). Order to drop: Layer 4 first, then Layer 3, never drop Layers 1-2.
- **artifact >16M:** strip code. Common spots: aggressive comments in build, redundant lever scaffolding.

### Task 13: 8×H100 three-seed validation

**Owner:** Claude Code, monitored by human
**Output:** 3-seed run of full stack on 8×H100.

**Steps:**
- Run seeds 42, 0, 1234 (matched to dexhunter's seeds for direct comparison)
- Each run captures all metrics
- Compute mean, std, per-seed deltas vs dexhunter's matched seeds

**Success:**
- 3-seed mean ≤ 1.0566
- Welch t-test vs dexhunter's distribution: p < 0.01
- All 3 seeds individually beat dexhunter's matched seed (a strong signal)

**Decision:**
- **All success criteria met:** proceed to PR submission (Task 14)
- **Mean ≤1.0566 but Welch fails:** add 4th and 5th seed. ($20 more.) Tighter variance estimate.
- **Mean 1.057-1.060:** doesn't beat #1797 by 0.005, but does beat #1797 outright. Submit as record candidate anyway and let organizers evaluate. Rule allows non-significant beats to be accepted as non-record submissions with novel methodology.
- **Mean >1.06157:** stack didn't beat #1797. Submit as non-record on Kevin's #1394 base with QK-Gain only.

### Task 14: PR submission package

**Owner:** human (Tanish, with Claude Code drafting)
**Output:** Complete PR ready for upstream review.

**Folder structure:**
```
records/track_10min_16mb/2026-04-XX_PR1797Base_QKGainSched_OptRot_AdamHD_LaCT/
├── README.md
├── submission.json
├── train_gpt.py                          (compressed wrapper for compute)
├── train_gpt_human.py                    (readable for review)
├── prepare_caseops_data.py               (inherited from upstream)
├── train_seed42.log
├── train_seed0.log
├── train_seed1234.log
├── ttt_seed42.log
├── ttt_seed0.log
├── ttt_seed1234.log
├── ablations/
│   ├── ablation_layer1_only.log
│   ├── ablation_layer2_only.log
│   ├── ablation_layer3_only.log
│   ├── ablation_layer4_only.log
│   └── ablation_summary.md
└── stats/
    ├── welch_test.txt
    └── seed_distribution.txt
```

**README must include:**
- Summary block with 3-seed mean, std, win bar comparison, p-value
- Head-to-head vs #1797 matched seeds
- "What this adds" section per layer with mechanism explanation and ablation
- Lineage: credit dexhunter (#1797), nprime06 (#1787), romeerp (#1729), samacqua (#1530), Kevin Clark (#1394), and the original idea credit for OptRot/AdamHD/LaCT papers
- Reproduction command
- Rule compliance checklist
- Honest "what didn't work" subsection if any layers were dropped

**submission.json fields (required by competition):**
- val_bpb, val_bpb_seeds, val_bpb_std
- val_loss_nats, val_loss_nats_std
- bytes_total
- github_id, name
- ttt_eval_time, train_time per seed

**Open the PR titled:** `Record: PR #1797 base + QK-Gain Schedule + OptRot + AdamHD + LaCT — val_bpb 1.0XXXX (3-seed mean)`

### Task 15: Parallel non-record submission

**Owner:** Claude Code, in parallel with Tasks 4-14
**Output:** Independent non-record PR documenting QK-Gain schedule on Kevin's #1394 base, 3-seed mean 1.07060.

**Why:** insurance. Even if the win path collapses entirely, you have a clean methodology submission with your validated 1×H100 numbers. This is the resume artifact regardless of the win outcome.

**Folder:** `records/track_10min_16mb/2026-04-24_SP8192_QKGainSchedule_NonRecord/`

This PR can be submitted at any time after Task 1 completes. **Do not block this PR on the win path.**

### Task 16: Parallel monitoring

**Owner:** human, every 12 hours
**Output:** Awareness of leaderboard movement.

**Steps:**
- Check https://github.com/openai/parameter-golf/pulls every 12 hours
- Filter for "Record:" titles, sort by newest
- If a PR with val_bpb < your current target lands, recompute the win bar:
  - new_bar = newest_score - 0.005
  - If new_bar < your projected stack output → adjust strategy
- Watch issue #140 for any organizer rulings (CaseOps legality, TTT legality, etc.)

### Task 17: Budget tracking

**Owner:** human

**Available budget:**
- $48 personal RunPod credit (current)
- $500 grant if it arrives (don't count on it)

**Estimated total spend (if everything runs once):**
- Tasks 1-3 (1×H100 reproduction): ~$30
- Tasks 4-11 (1×H100 layer integration runs): ~$60 (4 single-seed runs)
- Tasks 12-13 (8×H100 validation, 4 runs total): ~$40 community / $80 secure
- Buffer for debug retries: ~$30
- **Total: ~$160-200**

**Hard stop:** if cumulative spend exceeds $250 without 3-seed validation success, submit non-record and stop.

## Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| #1797 base doesn't reproduce on your build | Medium | High | Task 3 catches this before any layer is added. Fallback to PR #1787 base (1.06335). |
| Layers don't stack additively | High | Medium | Single-seed tests after each layer (Tasks 5, 7, 9, 11) catch this incrementally. Drop layers that don't help. |
| LaCT integration breaks PhasedTTT | Medium | Medium | Task 10 abort condition is explicit. Drop LaCT, ship Layers 1+2+3. |
| 8×H100 numerics differ from 1×H100 | Medium | High | Task 12 catches this. Two debug iterations allowed. |
| Someone lands sub-1.05 PR | Low | Critical | Task 16 monitoring. If new bar moves below your projected output, pivot to non-record immediately. |
| Compute runs out before validation | Low | High | Task 17 hard stop at $250. |
| CaseOps tokenizer ruled illegal | Low | Critical | Watch issue #140 daily. If ruled illegal, all CaseOps-based PRs invalidated and bar drops to 1.0810. Your 1.07060 result becomes new SOTA. |

## Open questions

1. Does the existing readable train_gpt_human.py for #1797 exist in their submission folder, or do we need to decompress it from their wrapper? (Affects Task 4 starting point.)
2. What's the exact OptRot rotation construction — Hadamard, Cayley, learned? Paper specifies; need to read carefully. (Affects Task 6 implementation complexity.)
3. Is AdamHD's Huber threshold per-parameter, per-layer, or global? (Affects Task 8 tuning surface.)
4. Does LaCT in the paper use the same Muon variant as the competition, or a different fast-weight optimizer? (Affects Task 10 integration depth.)

## Success metrics

**P0 (win):** 3-seed mean ≤ 1.0566 on 8×H100 with p<0.01 vs #1797. Submit record PR. Probability: 8-15%.

**P1 (placement):** 3-seed mean strictly below #1797 (≤1.06156) but failing the 0.005 significance bar. Submit as record candidate, accept non-record if downgraded. Probability: 25-35% (includes P0 cases).

**P2 (insurance):** Non-record submission with QK-Gain methodology landed on Kevin's #1394 base. Probability: >85%, achievable independent of all upstream tasks.

## What this PRD is consciously NOT optimizing for

- **Speed.** No timeline targets on tasks because rushing past abort conditions is how runs get wasted. Each single-seed test is a $15 information purchase. Pay it slowly.
- **Conservative ceiling.** Fewer layers = safer = lower ceiling. We're optimizing ceiling.
- **Tokenizer changes.** Multi-day implementation risk exceeds the calendar window.
- **Pre-eval TTT or memorization-style scores.** Explicitly ruled non-spirit.

## Decision authority

- **Tanish (human):** final call on all submissions, all spend over $50 in a single transaction, abort decisions on Tasks 12-14.
- **Claude Code:** everything in Tasks 2, 4, 6, 8, 10 (implementation) and Tasks 5, 7, 9, 11 (single-seed validation runs that cost ≤$20 each).
- **Claude (this conversation):** strategic guidance, status checks, mid-task pivots when ambiguous decisions arise.
