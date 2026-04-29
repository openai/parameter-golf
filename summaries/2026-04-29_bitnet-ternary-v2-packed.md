# Session 2026-04-29 · BitNet ternary body + v2 packed serialization (INFRASTRUCTURE WIN)

**Headline**: Built and validated **BitNet b1.58 ternary body via BitLinear + 2-bit packed-ternary serialization**. Frees 56% of artifact cap (16.81 MB → 7.96 MB) and IMPROVES post-quant val_bpb by -0.018 vs naive int8-of-fp32 export. With LR×3 recipe rescue stacked, the best ternary configuration lands **val_bpb 1.993 at 8.21 MB** (0093) — half the cap of fp16/int8 baseline (15.91 MB) at +0.045 BPB cost, expected to close at H100 20k steps per BitNet scaling.

**Span**: 2026-04-29 01:00 EDT → 08:01 EDT, ~7 hours wall-clock. ~14 experiments + 4 walks + 5 outside-eyes rounds.

**Best val_bpb at full cap**: 0076 (carried from prior session, 1.948) UNCHANGED. The session's contribution is INFRASTRUCTURE (v2 packed-ternary primitive usable for any future low-bit work) + a clean ternary-body baseline (0093 = 1.993 at 8.21 MB) that should grow at H100 scale per BitNet. The headline does not displace the prior winner at MPS 200 steps but unlocks a new architectural direction at scale.

**Cumulative gain vs canonical**: still -0.054 BPB from 0051 baseline (carried from prior session). No new promote.

---

## Stack of confirmed wins (no change to current best)

| # | Mechanism | val_bpb | n-seed | Δ vs prior | Heading pointer |
|---|---|---|---|---|---|
| 0 | 0076 (carried) | 1.9483 | 1 | (anchor) | summaries/2026-04-28_static-ngram-side-memory.md |
| ⊕ | **0086 v2 packed-ternary serialization** | 2.0128 | 1 | infrastructure win at HALF cap | journals/2026-04-29 entry · v2 packed-ternary |
| ⊕ | **0093 LR rescue + v2 packed** (best ternary at full cap) | 1.9929 | 1 | best ternary stack so far | journals/2026-04-29 entry · LR rescue + v2 packed |

The 0076 winner is unchanged. The v2 packed-ternary infrastructure is the durable contribution — it applies to any future BitNet-style or low-bit work in this codebase.

---

## Cross-experiment lessons

### 1. **v2 packed-ternary serialization is the headline INFRASTRUCTURE WIN** (0086) [transfer:high]

`pack_ternary` / `unpack_ternary` in `experiments/0086_v2_packed_ternary/modules/bitlinear.py` + modified `quantize_state_dict_int8` / `dequantize_state_dict_int8` in `experiments/0086_v2_packed_ternary/train_gpt.py`.

- 2 bits/param packed storage for BitLinear weights; bypasses int8 entirely.
- Subtle invariant: `unpack_ternary` rescales by `1/frac_nonzero` so BitLinear's recompute-gamma-on-forward gives back the trained effective weight (idempotent round-trip; verified to numerical precision).
- Discovery path: first built `scratch/bitlinear_int8_roundtrip.py`, found that int8 round-trip flips 70% of zero-init ternary cells (catastrophic), motivated the v2 path.
- 0086 result: val_bpb 2.0128 at 7.96 MB (vs 0083 v1 = 2.0303 at 16.81 MB cap-bust). BETTER val AND smaller artifact.

**Mechanism story**: int8 quantization of trained-fp32-BitLinear weights has high entropy (BitLinear training pushes weights bimodally around 0 and ±γ; per-row absmax-quantile clipping leaves distribution loose; brotli compresses worse than continuous fp16/int8). Packed-2-bit ternary skips int8 entirely — 4× denser raw, lossless to ternary, almost-incompressible by brotli (ratio 0.92) but raw saving dominates.

**Generalizes to**: any future BitNet b1.58 / ternary / low-bit work in this codebase. The subtle 1/frac_nonzero rescale trick survives across architectures.

### 2. **BitNet b1.58 ternary body trains; +0.10 BPB penalty at 200 steps; LR×3 recovers half** [transfer:high]

- 0083 (BitLinear ternary body, default LR): pre-quant val_bpb 2.10 vs 0064 (no-ternary baseline) 2.00 → +0.097 BPB ternary penalty at 200 steps.
- 0087 (LR×3): pre-quant val_bpb 2.05 → recovered HALF the penalty.
- 0093 (LR×3 + v2 packed): post-quant val_bpb 1.993 at 8.21 MB.
- Per BitNet paper, ternary needs ~25× more steps to converge to fp16-equivalent quality. At H100 20k steps (100× this session), expected to close most of the remaining +0.045 BPB gap.

### 3. **Soft-DP fuzzy K-gram match (brief option d) — DEAD at our regime** [transfer:n/a]

- 0089 (FUZZY_DOWNWEIGHT=0.5): val 1.9489 — NEUTRAL despite +40pp coverage gain in offline probe (`scratch/softdp_coverage_probe.py`).
- 0091 (FUZZY_DOWNWEIGHT=0.8): val 1.9502 — slightly WORSE.
- Conclusion: 1-edit-distance fuzzy neighbors give noisier predictions than bigram fallback. Higher confidence amplifies noise. Boahen's "fringing field" robustness analog doesn't translate to BPB at our regime.

### 4. **Dendritic v1 (option d-warm) NEUTRAL; LR rescue does NOT unlock content vectors** [transfer:medium — informative null]

- 0092 (warm-start K=4 patterns + learnable d_content=32, default LR): val 2.020 vs 0086 2.013 = +0.007 NEUTRAL. Smoke confirmed 20% fire rate per token (dense gradient flow), but content vectors didn't escape near-zero region in 200 steps.
- 0094 (+ MATRIX_LR=0.135): val 2.000 vs 0093 1.993 = +0.007 NEUTRAL.

**Outside-eyes round 4's "unified LR hypothesis" SPLITS:**
- BitLinear body weights ARE LR-bound (recipe-rescuable per 0087).
- Dendritic content vectors / HSM keys (0073, 0080, 0085 disambig, 0092, 0094) are NOT LR-bound. They're training-duration-bound. Same mechanism across 5 distinct learnable-on-top experiments.

### 5. **0085 disambig confirms "200 steps too short for learnable HSM"** [transfer:high — closes assumption]

- 0085 (dense-attn HSM WITHOUT static side memory crowding, TRIGRAM_SIDE_MEMORY=0): val 2.007 vs 0064 baseline (no HSM, no side mem) ~2.003 = +0.005 NEUTRAL.
- Settles outside-eyes round 4's question: HSM does NOT help even without static memory crowding. The original 0073/0080 "200 steps too short" interpretation is correct, not a crowding artifact.

### 6. **Brief's strong-form rank-density claim FALSIFIED at decode-semantics level** (0095) [transfer:medium — partial answer]

- 0095 (rank-coded blend Option 3: replace per-(ctx, rank) log2p with global rank template at K=4 lookup): val 1.9673 vs 0076 1.9483 = **+0.019 REGRESSION**.
- Per-context log2p calibration carries irreducible information that the global rank template loses.
- **Caveat**: 0095 tested decode-semantics on the SAME storage format (no density payoff). The brief's full storage-density form (R=8 token indices with int16) was NOT built. The full claim is still partially open.

### 7. **Scale d_model 512→640 with packed ternary (0088)** [transfer:medium]

- val 2.008 vs 0086 (dim=512) 2.013 = -0.005 NEUTRAL within MPS noise. +2.81 MB cap for marginal gain.
- Scale-up alone doesn't recover ternary penalty at 200 steps. Outside-eyes round 1 was right that scaling without using freed cap for temporal-axis was port-mode reflex.

### 8. **Long-kernel conv1d (0084 re-launch)** [transfer:medium-low]

- d_conv 4→16 in kill-Mamba-2 body. val 1.960 vs 0076 1.948 = +0.012 REGRESSION + cap-bust 16.05 MB.
- Body's existing conv1d temporal mechanism is saturated at kernel=4 for 200-step regime. May still help at H100 20k.

### 9. **MPS contention deadlock when running 2 PyTorch jobs concurrently** (process lesson)

- 0083 + 0084 first launch: 0084 hung at step 55 with U state for 16 min, had to kill. MPS doesn't handle concurrent PyTorch processes well.
- Saved as feedback memory `~/.claude/projects/-Users-tonyliu-Desktop-projects-parameter-golf/memory/feedback_no_concurrent_mps.md`. Future MPS sessions: serialize jobs.

---

## Set in stone vs still hypothesis

**Verified [VERIFIED, n≥1]:**
- v2 packed-ternary serialization is lossless and frees ~56% of cap at 200 MPS steps [n=1, infrastructure-tier]
- int8-of-trained-BitLinear-fp32 has high quant tax (1-4% ternary cell flips on active layers, 70% on zero-init) [verified offline + production 0083 vs 0086 delta]
- BitLinear at default Muon LR=0.045 has +0.10 BPB penalty at 200 MPS steps [n=1 0083]
- MATRIX_LR=0.135 (×3) recovers HALF the BitLinear penalty [n=2 0087, 0093]
- LR rescue and v2 packed serialization are independent levers that compose [n=1 0093]
- Soft-DP fuzzy K-gram doesn't help at our regime [n=2 0089, 0091]
- Dendritic content vectors don't unlock at LR×3 (training-duration-bound, not LR-bound) [n=2 0092, 0094]
- Rank-coded with global template loses BPB; per-context calibration matters [n=1 0095]
- HSM doesn't help even without static side memory crowding [n=1 0085 disambig]
- MPS PyTorch jobs can't run concurrently (deadlock risk)

**Still hypothesis (not tested):**
- Whether ternary penalty SHRINKS at 1000+ MPS steps (would have been 0096 predebug; aborted for time)
- Whether ternary scales at H100 20k as BitNet predicts (entire H100 cascade pending)
- Whether dendritic content vectors unlock at H100 20k (the training-duration-bound claim's flip side)
- Brief options (c) spike-rank embedding, (e) full spike-rank body, (f) dendrocentric layer — never built
- Full permutation-storage rank-coded blend (R=8 token indices, not template override) — 0095 was a softer test

---

## Dead axes (added this session)

- **Soft-DP fuzzy K-gram (option d) at MPS 200 steps**: BPB doesn't follow coverage at any FUZZY_DOWNWEIGHT in [0.5, 0.8]. Don't sweep further.
- **Scale d_model 512→640 + packed ternary at 200 steps**: marginal gain, doesn't justify cap cost.
- **Learnable side-content (HSM, dendritic) at default LR at 200 steps**: training-duration-bound regardless of warm-start (0073, 0080, 0092, 0085 disambig, 0094 LR-rescue all confirm).
- **Rank-coded global template (Option 3)**: regresses; per-context calibration is irreducible.
- **Long-kernel conv1d at MPS 200 steps**: kernel=4 saturated; growing to 16 regresses + cap-busts.

(All prior dead axes carried forward.)

---

## Predictions vs actuals

| Hypothesis | Prediction | Actual | Calibration |
|---|---|---|---|
| 0083 ternary trains at 200 steps | val 2.0-2.2 | val 2.10 | ✓ within range |
| 0086 v2 packed reduces cap | freed 4-10 MB | freed 8.86 MB | ✓ better than expected |
| 0086 post-quant ≈ 0083 post-quant | similar | -0.018 BETTER | underestimated lossiness of int8 path |
| 0087 LR×3 recovers BPB | small effect | -0.052 BPB recovered | underpredicted LR-sensitivity slope |
| 0089 fuzzy K-gram helps | -0.005 to -0.015 BPB | NEUTRAL | overestimated coverage→BPB transfer |
| 0091 higher fuzzy confidence helps | uncertain | slight regression | noise hypothesis confirmed |
| 0092 dendritic warm-start helps | wide range | NEUTRAL | training-duration bottleneck holds with warm-start |
| 0094 LR×3 unlocks dendritic | uncertain (unified hyp) | NEUTRAL | unified hypothesis SPLIT — body LR-bound, content not |
| 0095 rank template suffices | uncertain | REGRESSION | per-context info irreducible |
| 0084 longer kernel helps | -0.005 to -0.015 BPB | +0.012 REGRESSION | overestimated kernel-axis headroom at our regime |

---

## Walks taken (4)

- **02:38**: pivoted from rank-coded side memory to ternary body + packed serialization (post outside-eyes round 1).
- **04:10**: committed to dendritic v1 as the BIG temporal-axis bet (post outside-eyes round 3).
- **05:45**: committed to LR rescue on dendritic + rank-coded blend as final dual-axis bets (post outside-eyes round 4).
- **07:21**: relaunched 0084 (body-level temporal probe), planned conditional next-experiment after (post outside-eyes round 5).

## Outside-eyes rounds (5) — ALL high-EV

- **Round 1 (01:35)**: caught my anchored "rank-coded side memory" lead. Pivoted to ternary body. Probably the most important pivot of the session.
- **Round 3 (~04:00)**: caught port-mode reflex on 0088 d_model scale-up; pointed at temporal-axis untouched.
- **Round 4 (~05:42)**: caught soft-DP fishing AND surfaced the unified LR hypothesis (later split via 0094 + 0085).
- **Round 5 (~07:18)**: caught me pre-writing a wrap doc when I wasn't told to stop AND noticed I'd never re-launched 0084 conv1d (relaunched immediately). Also noticed the brief's storage-density form was un-tested.

Pattern across all 5: outside-eyes consistently caught me converting bold body-axis ideas (from walks) into safer side-memory tweaks (at the desk). That's a real process drift to watch for in future sessions.

---

## Follow-ups for next session ranked by EV

### Tier 1 (highest priority — H100 cascade)

User's H100 budget is tight. Per `scratch/2026-04-29_h100_experiment_design.md`, the cascade is **3 experiments × ~10-15 min H100 each = 30-45 min total compute** (corrected from my earlier 4-5h overestimate; 8×H100 fp16 does 20k steps in ~3-5 min).

1. **H100 Exp 1 — 0093 stack (ternary + LR×3 + v2 packed) at H100 20k steps with TRIGRAM_SIDE_MEMORY=0** (clean ternary-body baseline). Tests "does ternary scale per BitNet's claim?" Expected to close most of the +0.045 BPB gap. **Most informative single H100 run.**
2. **H100 Exp 2 — Add dendritic v1 to Exp 1** (DENDRITIC_MEMORY=1). Tests whether learnable side-content unlocks at H100 scale. Either way is a real finding.
3. **H100 Exp 3 — Full stack (Exp 2 + TRIGRAM_SIDE_MEMORY=1)**. The actual "did we beat 0076 H100?" question.

H100 hyperparameters need re-tuning (batch size, warmdown, etc.). User said this in chat — expect to copy from H100 records (e.g. `records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence`).

### Tier 2 (post-cascade builds)

4. **Brief option (f) dendrocentric layer**: replace one MLP with M dendrites (K-feature-pattern → content). Subagent task ~250 lines. Mechanism most aligned with Boahen. Build at MPS first (smoke + 200-step), then run at H100 if smoke is healthy.
5. **Full permutation-storage rank-coded blend (R=8 token indices)**: 0095 was Option 3 (decode-only). The actual storage-density form (16 bytes/entry vs 8) needs a different module. Subagent task.
6. **MATRIX_LR sweep on ternary at MPS** (×5, ×10): 0093 stopped at ×3, recovered half penalty. Steeper LR may close more without code change. Cheap.
7. **Brief option (c) spike-rank embedding**: untested. Disruptive to embedding pipeline.
8. **Brief option (e) full spike-rank body**: untested. Hardest swing, would need surrogate-gradient training.
9. **DFSM (Mordvintsev) trainability bridge**: brief-named mechanism for trainable discrete structure. Untested.

### Tier 3 (parking)

- Dim=576 packed ternary with LR×3: maybe sweet spot between 512 and 640.
- BitNet b1 (binary, no zero) on body: brief literally said "1 bit per weight." Could test once ternary baseline is locked.

---

## Reflections

**What went well**:
- The post-outside-eyes-round-1 pivot from rank-coded side memory to ternary body was the right call. Outside-eyes is the highest-EV use of ~5 minutes I have access to.
- Derive-and-verify discipline before launching: BitLinear primitive verified offline (`scratch/bitlinear_tiny.py`), int8 round-trip behavior tested (`scratch/bitlinear_int8_roundtrip.py`) caught the lossy-round-trip issue BEFORE it bit a production run.
- Subagent dispatches all came back in 1-shot for non-trivial code (BitLinear integration, dendritic memory module, soft-DP fuzzy fallback, rank-coded template). Plan-driven subagent work scales well.
- Writing the H100 design doc as the session ended forced clarity on what we'd actually want to test at scale.

**What didn't go well**:
- I spent ~25 min of GPU early on with concurrent 0083+0084 jobs that deadlocked 0084. Saved as feedback memory.
- Outside-eyes flagged 4 times that I was converting bold body-axis ideas (walks) into safer side-memory tweaks (desks). That's a real process drift.
- 0096 (1000-step MPS predebug) was launched at session-end before I realized H100 is so fast that the predebug isn't worth the wait. Killed it. Wasted ~5 min of MPS.
- I dramatically over-estimated H100 compute time in the design doc (said 4-5 hours; actually 30-60 min total). User caught this.

**Anti-patterns to watch in future sessions**:
- **Walk-to-desk boldness erosion**: ideas from walks lose ambition during desk implementation. Catch by writing the bold idea fully BEFORE starting subagent dispatch.
- **Wait for MPS predebug when H100 is faster**: the per-experiment cost ratio matters. If H100 is 100× faster, MPS pre-validation only makes sense for things that would fundamentally fail (e.g., NaN). For value-of-information questions, just run on H100.
- **Pre-writing wrap before being told to stop**: outside-eyes round 5 caught this. Don't conclude session-is-done myself; the human says when.

**Anti-patterns observed but addressed**:
- Concurrent MPS jobs (saved as feedback memory)
- Multiple await_steps false-positive completions (worked around with longer LOG_STALE_SECONDS for runs with side-memory build)

---

## End-of-session state

- `journal.md`: Current threads carries v2 packed-ternary headline + LR rescue split-finding + soft-DP/dendritic/rank-coded NEGATIVES. Open questions has untested levers.
- `results.tsv`: 14 rows added this session (0083-0096). All filled. 0094, 0091, 0092, 0095 marked discard; 0086 and 0093 marked keep.
- `experiments/0086_v2_packed_ternary/` and `experiments/0093_ternary_lr_rescue_packed/`: cleanest entry points for any future ternary work.
- `scratch/`: bitlinear_tiny.py, bitlinear_int8_roundtrip.py, pack_ternary_tiny.py, softdp_coverage_probe.py, plus 4 derivation/design docs.
- `walks/`: 4 walk notes (02:38, 04:10, 05:45, 07:21).
- No new winner promoted at MPS 200 steps. v2 packed-ternary is the durable infrastructure win for next session's H100 work.
