# Session 2026-04-28 · static N-gram side-memory + per-context α + confidence gate

**Headline**: NEW SSM-best 2-seed mean **val_bpb 1.95141** (cross-seed σ_pair=0.0061), Δ vs prior session winner 2.00503 = **-0.054 BPB**. Δ vs pure-attn baseline 2.08759 = **-0.137 BPB**. The thread-2 contribution: a static N-gram dictionary built from training data, packed in the int8 artifact, blended with model logits at inference.

**Architecture**: same as 2026-04-27 winner (K=3 L=3 + SwiGLU mlp=8 + triple-parallel ATTN || kill-Mamba-2 + no-BG) PLUS:
- Brotli compression of the int8 artifact (freed 1.74 MB cap)
- Combined K=3 (top_N=100K) + K=4 (top_N=200K) static N-gram side-memory packed as model buffers
- Per-context α blend weights (sigmoid of trigram entropy, clip [0.30, 0.85])
- Model-confidence gate (skip blend when model max log2p > -1.0)

Path: `winners/2026-04-28_confidence_gated_per_context_alpha_blend/`. Artifact 15.91 MB (88 KB safety under 16 MB cap).

**Δ vs canonical**: -0.567 BPB (canonical 2.521 → 1.951).

**Span**: 2026-04-28 00:34 EDT → 2026-04-28 ~08:30 EDT, ~8h wall-clock.

**Theme**: thread 2 surprise — static N-gram side memory is a regime-specific training-shortcut prior at the under-trained 200-step MPS smoke. The model's BPB is close to the trigram's BPB at that regime, so blending recovers complementary information.

---

## Stack of confirmed wins (cumulative path canonical → current best)

| # | Mechanism | val_bpb | n-seed | Δ vs prior | Heading pointer |
|---|---|---|---|---|---|
| 0 | 0051 family triple-parallel kill-Mamba-2 (BASELINE) | 2.00503 | 4 | (anchor) | summaries/2026-04-27_kill_mamba2_cross_class.md |
| 1 | + Brotli artifact compression | ~2.0030 | 1 | ~0 (lossless) | 0064 entry · brotli_swap |
| 2 | + Combined K=3+K=4 static side memory blended at inference | 1.95990 | 2 | -0.045 | 0069/0072 PROMOTE entry · combined K3 K4 |
| 3 | + Per-context α blend weights (entropy-derived) | ~1.957 | 2 | -0.003 | 0074/0075 entry (kept, not separately promoted) |
| 4 | + Model-confidence gate (skip blend at high-confidence tokens) | **1.95141** | **2** | -0.005 | 0076/0077 PROMOTE entry · confidence-gated blend |

**Total compounded Δ vs baseline: -0.054 BPB** (additive on numbered levers; per-context α didn't fully manifest at 2-seed but the gate-stack did).

---

## Cross-experiment lessons (numbered, with journal pointers)

### 1. **Static trigram alone hits BPB 2.024 at our val cap — model is mostly memorizing N-gram statistics at 200 steps** (UU#6 entry · BIG FINDING)
A pure-Python K=3 backoff predictor built from 100M training tokens (no model at all) gets val_bpb 2.024 vs our model's 2.005. The 200-step MPS smoke is a regime where local statistics dominate. **This reframes most of our SSM-architecture deltas**: -0.083 BPB pure-attn → triple-parallel is largely a measure of how efficiently each architecture absorbs trigram-ish structure under under-training. The brief's "SSM recall gap" (Zoology) is about >3-gram associative recall — manifests at H100 20k-step, not at MPS smoke.

### 2. **Model + trigram blend at α=0.5 gives -0.088 BPB inference-time gain** (BLEND PROBE entry · OUTSIZED FINDING)
Per-token analysis: model and trigram have substantially orthogonal predictions. At α=0.5, blended val_bpb 1.9072 (vs model 1.9956). The structural reason: ~6% of val tokens have model log2p < -10 (very wrong); the trigram captures these with log2p ~ -1 to -2. The blend rescues those tokens by orders of magnitude per-token. Bounded downside (-0.51 bits/token max = log₂(α=0.7)), unbounded upside.

### 3. **Static N-gram side memory CAN be packed into the int8 artifact** (0067/0068/0069 entry)
The Mamba-2 family's int8 quantization passthrough preserves integer buffers cleanly. Packing 100K K=3 + 200K K=4 contexts as `(keys: int32, offsets: int32, next_ids: int16, log2p_quant: int8)` adds ~5.5 MB raw, ~2.3 MB brotli'd. Total artifact 15.7 MB within the 16 MB cap.

### 4. **K=4 contributes complementary info to K=3** (combined K3+K4 entry)
3-way blend `model + K=3 + K=4` at weights (0.7, 0.10, 0.20) gives -0.046 BPB, beating K=4 alone (-0.041). Even at heavy K=4 pruning (top_N=200K out of 4M raw), the K=3 contribution is non-zero. Higher K alone doesn't help statically (K-sweep entry), but blended with model and other K's it adds.

### 5. **Per-context α from trigram entropy: offline -0.014, production -0.005** (0074 entry)
Sigmoid map of per-context entropy with τ=0.5, threshold=3.0, clip [0.30, 0.85] gave a clear offline win. End-to-end production gain was smaller, attributed to int8 α quantization + cross-seed model variance. The σ widened from 0.005 → 0.010 at the 2-seed family. **A 65% offline→production gap** that the next session should diagnose if it pursues offline-tuned blend variants.

### 6. **Confidence gate stacks cleanly on per-context α** (0076 entry · PROMOTE)
The per-token analysis showed blend slightly hurts ~12% of tokens where model is confident (model log2p > -1). Skipping the blend on those tokens reclaims that small loss. -0.004 BPB on top of per-context α at 2-seed precision.

### 7. **HSM (hash-based learnable side memory) doesn't converge in 200 steps** (0073 entry · clean negative)
32 buckets × 5-bit hash + learnable value bank (zero init) added on top of 0069: val 1.9599 = NEUTRAL/SLIGHTLY-WORSE. UU#4 said hidden states have PR=5.4 (low rank) → 32 buckets should differentiate. Failure mode: gradient sparsity (each bucket value gets ~750 updates × dim, but the per-element updates are sparse). At H100 20k-step (100× more updates) HSM should work. Clean negative result that strengthens the "static side-memory works because it doesn't need training" thesis.

### 8. **Train-time blend (0071) crashed at production shape on MPS** (0071 entry · crash)
Bounds error in `trigram_blend_loss` at B=3, L=1024 — smoke at (B=4, L=256) on CPU passed. The MPS-shape-dependent bug is unresolved. The mechanism (model adapts to be complementary to the static prior during training) is interesting and untested. **Future-session priority: fix the bug, test the mechanism.**

### 9. **EMA doesn't stack with the side-memory blend at this regime** (0078/0079 entry)
β=0.999 catastrophic at 200 steps (window 1000 >> training duration → shadow stays mostly initial random weights → val 5.87). β=0.95 (window ~20) gives val 1.9533 = neutral on top of 0076. The static side-memory likely already absorbs the "smoothing" benefit EMA would provide. Single-seed; EMA is well-validated in the literature, no need for SEED=42 confirm.

### 10. **Hidden states are low-rank — empirically validates Boahen** (UU#4 entry)
Forward pass through 0051 winner: PR drops from 40.8 (block 0) to 5.4 (final block). Rank@90% var = 77 / 512 at final. This motivates the brief's (d) candidate: a SMALL key bank (16 keys × full d_model with attention readout) should saturate the natural rank and might escape the gradient sparsity that broke 0073's hash-based HSM.

---

## Dead axes (verified — don't re-test without changing other levers)

- **Parallel residual lanes** (0066): NULL on MPS regime. The H100 record's -0.0022 gain didn't transfer at 200-step. Possibly the routing params are under-trained, or our looped-topology constraint (per-unique-block routing, shared across loops) loses the per-layer flexibility that worked at H100. [0066 entry]
- **Asymmetric topology** (0065): position 0 demoted from parallel to pure kill-Mamba-2. NEUTRAL on val_bpb (Δ +0.0024) but freed 0.5 MB cap. Useful for future cap-stacking but not a win on its own. [0065 entry]
- **EMA at 200-step** (0078/0079): doesn't help on top of the side memory. Side memory already absorbs the smoothing benefit.
- **HSM with hash-based 32-bucket discrete keys** (0073): doesn't converge in 200 steps. Use dense-attention 16-key variant instead next session.
- **Per-context α with α_min=0.5, α_max=0.95**: too conservative; offline best is α_min=0.30, α_max=0.85.

---

## Set in stone vs still hypothesis

**Set in stone** (multi-seed + verified by analysis):
- Combined K=3+K=4 static side memory blended at inference gives -0.045 BPB (2-seed, 0069/0072).
- Confidence-gated per-context α stack gives -0.054 BPB cumulative vs 0051 family (2-seed, 0076/0077).
- Brotli q=11 saves 1.74 MB vs zlib level=9 on int8-quantized weights.
- Mechanism: blend rescues tokens where model is very wrong by orders of magnitude per-token, costs at most -0.51 bits/token where model is right (per-token analysis).
- Hidden states have PR ~5 at final block (UU#4).

**Still hypothesis** (single-seed or extrapolated):
- HSM with dense 16-key attention readout would work (extrapolated from UU#4).
- Train-time blend (0071) would help if the bug were fixed (extrapolated from offline analysis).
- AR self-gen GPTQ int6 would compose with side memory (extrapolated from H100 record).
- H100 transfer of side-memory: estimated -0.01 to -0.02 BPB. Not tested.

---

## Predictions vs actuals

No formal predictions table at session start. Implicit predictions tracked per-experiment in plan.md files.

Calibration check on key predictions:
- **0064 brotli**: predicted -1.0 to -1.5 MB cap saving. Actual: -1.74 MB. Close.
- **0068 K=4 alone**: predicted Δ -0.041 from offline. Actual: -0.041. Exact.
- **0069 combined K=3+K=4**: predicted Δ -0.045 from offline. Actual: -0.046. Exact.
- **0073 HSM**: predicted ~50/50 hypothesis (A) vs (B). Actual: hypothesis (B) — 200 steps insufficient.
- **0074 per-context α**: predicted -0.014 from offline. Actual: -0.005 single-seed, -0.003 at 2-seed. **65% gap** — calibration item for next session.
- **0076 confidence gate**: predicted -0.004 from offline. Actual: -0.0038 single-seed. Close.
- **0078 EMA β=0.999**: predicted -0.003. Actual: catastrophic +3.87 (forgot β/window math at short training).

Lesson: offline blend predictions are accurate when the cached model probs match production. Where they diverge (per-context α), the gap can be ~65%. Future offline-driven mechanisms should expect ~30-50% production erosion until proven otherwise.

---

## Walk reflections (consolidating walks/2026-04-28_0745.md)

**[WORTH_TESTING] Dense-attention HSM**: 16 learnable keys × 512 d_model with softmax attention readout (NOT hash-based). UU#4's PR=5.4 says natural rank is ~5; 16 keys saturates with attention mixing. Every key gets gradient on every token. Cap ~10 KB. The (d) candidate from the brief with empirical motivation that 0073 lacked. ~80 lines subagent + 30 min run.

**[SPECULATIVE] Side memory as training-shortcut prior**: the right reframe. Not "recall mechanism" but "embedded dataset statistics." Suggests other priors (sub-token co-occurrence, document-type clusters, distilled sub-vocab embeddings) may stack similarly. New direction.

**[WORTH_DERIVING] Cap-multiplier from AR int6**: int6 saves ~25% per layer (~3 MB at our int8 baseline) → frees cap that could double K=4 contexts. Could land ~1.94 BPB at the same cap as 0076 with much bigger side memory.

**[SPECULATIVE] 0071's MPS bounds bug fix unlocks train-time blend**: model adapts to be complementary to the static side memory during training. Genuinely under-explored.

---

## Follow-ups for next session ranked by EV

1. **Dense-attention HSM** (~1.5h subagent + run + analyze). UU#4-motivated, escapes 0073's gradient-sparsity failure mode. The brief's (d) candidate properly executed. Predicted -0.005 to -0.015 BPB.
2. **AR self-gen GPTQ int6** (~3h subagent task). Cap-multiplier on side memory. Frees ~3 MB → grow K=4 to top_N=400K → predicted -0.005 BPB on top of 0076 PLUS smaller artifact for H100.
3. **4-seed sentinel of 0076/0077** (~1h, 2 more seeds). Tightens σ_pair before further stacking. Reviewer flagged σ widening 0.003 → 0.005 → 0.006 across the stack.
4. **Train-time blend bug fix (0071)** (~1.5h debug + retry). Resolves the most under-explored thread-2 mechanism. The MPS bounds bug in `trigram_blend_loss` at B=3, L=1024 hides on CPU.
5. **(e)/(f) bold candidates from brief**: full spike-rank body, dendrocentric layer. Massive code changes. Best in non-record-track if needed.
6. **Higher-K static side memory with hashing** (~2h). K=5 / K=6 with hash buckets — speculative; might hit storage-vs-coverage tradeoff that K=4 200K hits well.
7. **Mini-depth-recurrence + side memory stack** (thread-1 deferred). RECUR_LAYERS=4,5 + REPEAT_UNTIE_MLP. Potential cap-savings.

---

## Reflections — what went well, what didn't

**Went well**:
- The static N-gram probe (UU#6) was the highest-EV CPU experiment of the session. ~5 min of code, produced a finding that reshaped everything after.
- Per-token analysis (12:42 EDT entry) is paper-quality mechanism story. Bounded downside, structural asymmetry, clean explanation of why the blend works. Next session should preserve and extend this kind of analysis.
- Subagent-handoff worked cleanly for all six dispatched code changes (0066, 0067, 0068, 0069, 0073, 0074, 0076, 0078). All landed SMOKE OK at production-shape MPS after the 0071 lesson was learned.
- Outside-eyes review at 07:30 was load-bearing — sharpened the diagnosis that the session drilled UU#6 instead of exploring the brief's bold candidates.

**Didn't go well — anti-patterns to break next session**:
- **Drilled UU#6 axis for 6 experiments** (0067, 0068, 0069, 0074, 0076, 0078). The brief explicitly named this as the anti-pattern. The bold (d), (e), (f) candidates remained untouched. Outside-eyes flagged it correctly.
- **σ widened with stacking** without 4-seed sentinel of 0076. Promote at +0.0085 sits at ~1.4σ at 2-seed precision. Should have re-sentinel'd before stacking 0076 on 0069.
- **0078 EMA β=0.999 was a math error**, not a genuine experiment. Should have computed effective window = 1/(1-β) = 1000 vs 200 step training BEFORE launching. Cost 30 min.
- **Did SEED=42 confirms on every variant** (0072, 0075, 0077) when single-seed was sufficient for known levers (per user feedback at hour 7). Burned ~1.5h that could have gone to bold candidates. Saved as `feedback_seed_confirm_discipline.md`.
- **Spawned outside-eyes too late** (hour 7 of 8). Reviewer noted: should have invoked by experiment 4 of any axis. Self-confirming chains across walks/research are the most common failure mode; the antidote is a fresh subagent reading earlier.

**Calibration learnings**:
- Offline blend predictions: accurate within ±0.001 when using cached model probs (0068, 0069). Up to 65% optimistic when extrapolating to mechanisms that depend on per-element quantization (0074 per-context α).
- HSM gradient sparsity at hash-based discrete keys was foreseeable but not foreseen. The math: 32 buckets × ~24K tokens/batch = 750 grad updates per bucket value PER DIMENSION × 200 batches = 150K micro-updates per element BUT spread across many ID combinations. Should have computed before launching.

---

## Hand-off

**Immediate next move**: **Dispatch dense-attention HSM as 0080.** 16 learnable keys × 512 d_model with softmax attention readout, added after the last block before final norm. Init keys with k-means on training-data hidden states (one shot). Trains end-to-end with the model. Cap ~10 KB. The brief's (d) candidate with empirical motivation. If it works (BPB < 1.95 at single-seed), it's the second mechanism on top of the static side memory and a much cleaner story than further blend tuning.

**Do not detour into more side-memory blend variants without first running 0080.** The reviewer's diagnosis: σ has widened, marginal Δ has shrunk, the axis has been mined. The next experiment must be a different axis or a much bigger swing.
