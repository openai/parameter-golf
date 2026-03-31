# Lab Pipeline — Ranked Hypothesis Queue

**Updated:** 2026-03-31
**Crawler champion:** 1.18672385 BPB · 8.61MB · `crawler/2026-03-29_BW5/`
**Neural champion:** 1.10986874 BPB · 15.44MB · `neural/2026-03-30_Rascal_II/`

Ranked by estimated potential impact. One variable per test, always. Gate before 8x, always.

---

## TIER 1 — Leaderboard-threatening

These have claimed or theorized BPB deltas large enough to change standings.

### [NEURAL] SLOT — Sample-Specific Eval Adaptation
**Status:** Designed. Shelved pending torch 2.11 pod.
**Variable:** `SLOT_ENABLED=1` (eval-side only, training unchanged)
**Mechanism:** At eval time, for each sliding window batch: freeze hidden states, optimize a small additive delta for 8 AdamW steps on the LM loss, score with the adapted delta. Model weights never modified.
**Claimed delta:** ~−0.021 BPB (arXiv:2505.12392v2). If real, that's leaderboard-smashing.
**Legal:** Yes — score-first, self-supervised, no external labels.
**Cost:** ~$0.50 gate (1GPU, 2000 steps), ~$3-4 full run.
**Prerequisite:** torch 2.11 pod. Fix `experiments/QK_GAIN_SLOT_Gate/` REPO_ROOT first.
**Risk:** Proxy result may inflate. The claimed delta is from a different codebase.

---

### [NEURAL] QK_GAIN_INIT=4.0 — Sharper Initial Attention Focus
**Status:** Designed. Shelved pending torch 2.11 pod.
**Variable:** `QK_GAIN_INIT=4.0` (default 1.5). Zero code change.
**Mechanism:** `q_gain` per-head scalar initialized at 4.0. Model is free to train away — this is an init effect, not a constraint. Drives sharper early attention gradients.
**Claimed delta:** ~−0.006 BPB. Source: external, 45 runs across 3 codebases.
**Cost:** Included in existing SLOT gate. Two-for-one test.
**Prerequisite:** Same torch 2.11 pod as SLOT.
**Risk:** Init effects shrink as training progresses. May wash out at full run.

---

### [CRAWLER] Delta Anchor / Delta Farce (BDF series)
**Status:** Designed (memory). Not yet scripted.
**Variable:** Per-loop dynamic causal state vector at loop boundaries.
**Mechanism:** Battery (9,1,1) differentiates *reading* — each loop attends at a different causal horizon. Delta anchor completes the pair: differentiates *writing*. Each loop commits a small learned anchor state (dim ~32) for the next loop to condition on, instead of all loops writing blindly into the same residual stream. Extends FLOW (inst_dim) from static identity bias → dynamic per-loop time state.
**Why high impact:** The current crawler has no dynamic causal memory crossing loop boundaries. Loop 1 cannot distinguish what loop 0 extracted causally from what was already in the residual. This is the fundamental architectural gap. Battery addressed the attention side. This addresses the output side.
**Arm structure (BDF series):** BDF-00 control · BDF-01 anchor_dim=32 loop→loop · BDF-02 anchor_dim=64 · BDF-03 anchor_dim=32 symmetric · BDF-04 anchor_dim=32 + seeded from tap
**Estimated delta:** Unknown — first principled test of this gap. Could be small or could be step-change.
**Cost:** Gate ~$0.50 per arm, 4-5 arms.
**Prerequisite:** None on BW5. Can run now. Does not require cannon or skipgram to confirm first.
**Risk:** High complexity. Could introduce instability. Could also be that battery+residual already routes causality well enough.

---

### [CRAWLER] Tap (BWT series) — Per-Loop Gated Encoder Tap
**Status:** Designed (hypothesis in junkyard). Param already exists in BW5 (`CRAWLER_TAP_DIM`).
**Variable:** `CRAWLER_TAP_DIM=32` (default 0 = disabled). `CRAWLER_TAP_LOOP_SPECIFIC=1`.
**Mechanism:** Project intermediate encoder layer outputs (shallow + deep) once into a small tap_dim embedding. Each crawler loop injects a learned projection of these frozen encoder signals into its residual. Gives the crawler a stable, pre-quantization anchor to check against as it loops. The tap signal is computed once before looping — negligible overhead.
**Why high impact:** The crawler accumulates quantization error across 3 loops with no stable reference. FLOW is self-referential (tracks its own drift). The encoder tap provides an uncontaminated anchor — the pre-loop signal the crawler is supposed to be refining. This directly attacks the quant gap.
**Arm sweep:** tap_dim ∈ {16, 32, 64} · per-loop vs shared · shallow/deep/all encoder layers
**Estimated delta:** Medium-large. Quant gap reduction is the primary lever for int6_sw_bpb.
**Cost:** ~$3-4 full run after gate. Multiple gate arms.
**Prerequisite:** None — CRAWLER_TAP_DIM=0 is already the BW5 baseline.
**Risk:** Tap projection cost adds latency. Per-loop specificity adds params. May need careful tuning.

---

## TIER 2 — Clear signal, lower complexity

### [NEURAL] Trigram on Rascal II
**Status:** Code already in vault. Just needs `TRIGRAM=1` env var.
**Variable:** `TRIGRAM=1` (default 0). Zero extra parameters.
**Mechanism:** Identical to BW6_Skipgram: trigram hash `(t-2, t-1, t)` added into same 2048-slot bigram embedding table. Neural SOTA already has the implementation — it just defaults off.
**Estimated delta:** Small-medium. If skipgram helps crawler, likely helps neural too (same mechanism).
**Cost:** ~$0.50 gate.
**Prerequisite:** None. Run standalone on Rascal II base.
**Risk:** Low. Zero params, warm start, additive.

---

### [CRAWLER] Shared Flat Layer Weights
**Status:** Concept. Not scripted.
**Variable:** Cross-block weight tying in the 4 flat U-Net encoder/decoder layers.
**Mechanism:** The 4 flat layers (2 encoder + 2 decoder) currently have unique weights. If encoder layer pairs or encoder/decoder symmetric pairs share weights, that frees substantial parameter budget (~4M params at dim=512). Those freed params can be reinvested into the crawler block (wider MLP, deeper tap, etc.).
**H8 finding (neural track):** Weight-shared depth tested vs unique layers — crawler loops already demonstrate this pattern works. Flat layer sharing is unexplored on the current config.
**Estimated delta:** Unknown. Could be neutral (weight tying at this scale doesn't hurt) with free budget to reinvest, or could be negative.
**Cost:** ~$0.50 gate.
**Prerequisite:** None. Run as standalone gate vs BW5.
**Risk:** Medium. Flat layers serve distinct encoder/decoder roles; tying them may hurt.

---

### [CRAWLER] BW6_Skipgram (Trigram)
**Status:** 8GPU gate running now.
**Variable:** `TRIGRAM=1`. Zero extra parameters.
**Mechanism:** Trigram hash `(t-2, t-1, t)` added to existing bigram embedding. Same 2048-slot table, same projection. Richer n-gram input context for the crawler loops at zero cost.
**Estimated delta:** Small-medium. Zero-param enrichment.
**Gate pass criterion:** BW6SK-01 raw_bpb < control AND step_avg ±2ms.

---

### [CRAWLER] Smear Gate
**Status:** Designed. Param already exists (`CRAWLER_LOOP_SMEAR=0`). Flip to 1.
**Variable:** `CRAWLER_LOOP_SMEAR=1`. ~512 learned scalars.
**Mechanism:** Learnable sigmoid blend between consecutive loop outputs (current loop output ↔ previous loop output). Zero matmuls — essentially free compute. Soft low-pass filter across loop depth. Damps quantization error amplification across loops (each loop re-processes the previous loop's error through the same int8 weights).
**Estimated delta:** Small. The error damping effect is real but limited.
**Cost:** ~$0.50 gate. Trivial.
**Prerequisite:** None. Add to BW5 gate.
**Risk:** Very low. Zero-init gate → sigmoid(0)=0.5, model learns direction. Warm start.

---

### [CRAWLER] BW5_Cannon Full Run
**Status:** run.sh ready. Gate passed (74.81ms vs 74.84ms, −0.00016 int6_sw_bpb).
**Variable:** `CRAWLER_CANNON_TYPE=scalar`. 3 params.
**Signal at gate:** Tiny speed gain, tiny quality gain, +343KB size regression.
**Unknown:** Whether the quality signal compounds over 8000 steps or stays marginal.
**Cost:** ~$3-4.

---

## TIER 3 — Refinement / cleanup

### [CRAWLER] Pyramid Small Choke (dim=128 or 256)
**Status:** Concept. Derived from pyramid failure post-mortem.
**Variable:** `CRAWLER_MLP_CHOKE_DIM=128` (or 256). Shape=pyramid.
**Why:** Pyramid failed at dim=512 because 1.57M cold params compound training burden over time. Smaller choke = less burden. The structural idea is not wrong — just needs a feasible parameter count.
**Estimated delta:** Small-medium if the concept holds at smaller scale.
**Additional variants:** Warm initialization of bottleneck weights; dedicated LR schedule for choke layers.
**Prerequisite:** None. But learn from pyramid failure — run gate at 2000 steps, not 500.

---

### [CRAWLER] XSA Coverage Sweep on BW5
**Status:** Concept. Pre-BW5 tests showed XSA coverage is a quant-robustness lever.
**Variable:** `XSA_LAST_N=13` or `=15` (current: 11, ceiling: 15 for 15-block model).
**Context:** XSA=11 was tuned pre-fullgraph. BW5's compile optimizations may have changed the step-time headroom. Full coverage (XSA=15) adds overhead but may return quant gap reduction.
**Estimated delta:** Small. Pre-BW5 the gain was real but marginal vs step cost.
**Risk:** Speed regression. Measure step_avg carefully.

---

### [CRAWLER] Warmdown Tuning
**Status:** Low priority. BW5 seed gap.
**Variable:** `WARMDOWN_ITERS` or LR taper shape.
**Why:** BW5 seed=300 is +0.00012 worse than Leg 3 seed=300. Mean is better but seed=300 doesn't individually confirm. Closing this gap makes the champion more robust.
**Estimated delta:** Tiny. This is seed-gap management, not a quality leap.

---

### [NEURAL] QAT Tuning
**Status:** Not started.
**Variables:** `LATE_QAT_THRESHOLD` (current 0.15), QAT start step (current ~6070).
**Why:** The quant gap (roundtrip vs sliding window) in Rascal II is ~0.001. Earlier/stronger QAT may tighten it. Risk: too-early QAT disrupts training.
**Estimated delta:** Small. Rascal II is already near-optimal.

---

### [NEURAL] Architecture Capacity
**Status:** Not started.
**Variables:** `BIGRAM_VOCAB_SIZE=4096`, `ROPE_DIMS=32`, extra XSA layer.
**Context:** 0.5MB headroom under 16MB cap. Any expansion risks the size gate.
**Estimated delta:** Unknown. High risk given tight size constraint.
**Prerequisite:** Any candidate must pass `bash submissions/validate.sh` size check.

---

## Combined / Downstream (after individual validation)

| Combo | Prerequisites | Notes |
|-------|--------------|-------|
| Cannon + Skipgram | Both individually promote | Two-variable test |
| Cannon + Smear | Both individually promote | Likely compatible |
| Tap + Delta Anchor | Both individually gate | Would be BW7+ architecture |
| SLOT + Trigram (neural) | Both individually gate | Eval + training enrichment |

---

## Shelved (needs environment fix)

| Experiment | Location | Blocker |
|-----------|----------|---------|
| QK_GAIN + SLOT gate | `experiments/QK_GAIN_SLOT_Gate/` | Needs torch 2.11. REPO_ROOT path also broken in run script. |
| QK_SLOT (neural) | `junkyard/experiments/QK_SLOT_Ablation/` | Same torch 2.11 issue. Pod ran at 3358ms/step (4× slow). |
