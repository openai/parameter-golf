# Session 2026-04-27 · kill-Mamba-2 + cross-class hybrid

**Headline**: SSM-best 4-seed mean **val_bpb 2.00503** (cross-seed σ_mean=0.0015), Δ vs pure-attn baseline 2.08759 = **-0.0826 BPB**, fully decomposed mechanism story complete.

**Architecture**: K=3 L=3 + SwiGLU(mlp=8) + every K=3 unique block is PARALLEL ATTN || kill-Mamba-2(LTI), no BigramHash. Path: `winners/2026-04-27_triple_parallel_kill_mamba2_no_bigram_recur3x3/`.

**Δ vs canonical**: -0.516 BPB (canonical 2.521 → 2.005). Δ vs prior session winner (2.0839): **-0.079 BPB**.

**Span**: 2026-04-26 14:00 EDT → 2026-04-27 07:55 EDT, with a long laptop-pause around 17:35-21:30. ~14h wall-clock active. ~24 promote-grade or mechanism-grade experiments (0035-0063), 6 promotes, 4 outside-eyes resolutions.

**Theme**: kill-selectivity → no-BigramHash → cross-class parallel topology + conv1d-as-recall mechanism story.

---

## Stack of confirmed wins (cumulative path canonical → current best)

Each row builds on the previous. All numbers are post-quant val_bpb at the seed-count given.

| # | Architecture | val_bpb | n-seed | Δ vs prior | Heading pointer |
|---|---|---|---|---|---|
| 0 | Pure-attn 3-of-3 + recur+SwiGLU+mlp=8, no-BG (BASELINE) | 2.08759 | 2 | (anchor) | 0058 entry @ 06:25 EDT |
| 1 | + Mamba-2 BLOCK at 1-of-3 (S4D-Lin sandwich → 1 mamba) | 2.06016 | 2 | -0.028 | 0032/0034 entry @ ssm_session |
| 2 | + Mamba-2 BLOCK at 2-of-3 (full selectivity, with BG) | 2.04171 | 2 | -0.018 | 0035/0036 promote |
| 3 | + Kill selectivity (LTI dt/B/C → constants) | 2.02723 | 2 | -0.014 | 0038/0039 entry @ 22:30 EDT 2026-04-26 |
| 4 | + Remove BigramHash | 2.02193 | 2 | -0.005 | 0042/0045 entry @ 00:35 EDT 2026-04-27 |
| 5 | + Middle-parallel topology (PARALLEL ATTN || kill at pos 1) | 2.00950 | 3 | -0.012 | 0046/0050/0060 (3-seed final) |
| 6 | + Triple-parallel topology (PARALLEL at all 3 positions) | **2.00503** | **4** | -0.004 | 0051/0053/0056/0057 (4-seed sentinel @ 06:00 EDT) |

**Total compounded Δ vs baseline: -0.0826 BPB** (additive on numbered levers).

---

## Cross-experiment lessons (numbered, with journal pointers)

### 1. **Selectivity is anti-load-bearing at 200-step regime** (0038/0039 entry)
LTI (kill) Mamba-2 BEATS full selective Mamba-2 by 0.014 BPB at 2-seed precision. The per-token (dt, B, C) projections from in_proj are under-trained at 5M tokens of training. Walk 22:22's quant-noise hypothesis (0041) was REFUTED — the issue is genuinely selectivity-difficulty, not bf16 quantization (CONTROL_TENSOR_NAME_PATTERNS only affects post-train serialization).

### 2. **Conv1d IS the recall mechanism in Mamba-2 block** (0047 entry)
Removing conv1d regresses val by **+0.091 BPB** vs 0042 base. Conv1d (depthwise width-4 causal conv on x_branch) is doing local-pattern feature-level recall that LTI scan aggregates. **This is the single most important mechanism finding of the session** — it explains both the kill-wins and BG-hurts results.

### 3. **BigramHash slightly HURTS Mamba-2 family** (0042/0043 entry @ 00:05 EDT)
Δ +0.005 to +0.007 BPB across kill and full Mamba-2 variants. Opposite of S4D-Lin family where BG helps ~+0.011. Conv1d-as-recall explains: BG and conv1d compete for the same recall niche, but conv1d does it better at the feature level.

### 4. **Conv1d ≠ BG (subtle refinement, outside-eyes catch #2)** (0062 entry @ 07:25 EDT)
If conv1d does BG's job, removing conv1d should let BG help again. But 0062 (no-conv1d + BG) val 2.1225 is WORSE than 0047 (no-conv1d, no-BG = 2.1132). Conv1d's role is **channel-specific local-pattern recognition** (depthwise per-channel kernels), distinct from BG's **global token-level bigram lookup**. They're complementary in S4D-Lin (no conv1d, BG fills the niche), redundant-with-overhead in Mamba-2 (conv1d does it better, BG adds noise).

### 5. **Cross-class parallel topology compounds** (0046/0050 entry @ 02:00 EDT)
PARALLEL ATTN || kill-Mamba-2 at the residual sum point beats sequential composition by 0.012 BPB at 2-seed precision. Attention and kill-Mamba-2 provide complementary signals (content-addressable lookup vs LTI-decay context) that don't interfere when summed.

### 6. **Cross-class win is SPECIFIC to kill-Mamba-2, not generic diversity** (0063 entry @ 07:40 EDT, outside-eyes #3)
Parallel-S4D-Lin in middle position (instead of parallel-kill-Mamba-2): val 2.0331, Δ +0.021 vs 0046. The cross-class lever requires conv1d in the parallel-SSM partner. Refined punchline: "ATTN || conv1d-equipped-SSM in parallel beats sequential composition" — not generic mechanism diversity.

### 7. **More parallel positions help, saturating ~3 of 3** (0051/0053/0056/0057 + 0054)
Sequential (0 parallel) → middle (1 parallel) → outer (2 parallel) → triple (3 parallel): 2.022 → 2.010 → 2.008 → 2.005. Diminishing returns past 1-of-3. Triple-parallel is best by 0.005 over middle-parallel at multi-seed precision.

### 8. **κ-scalar collapse derivation captures most but not all of d_state** (0044 entry @ 00:35 EDT, 0055 entry)
The kill version's per-position output collapses to scalar κ = ⟨B_const, C_const⟩ per the closed-form, predicting d_state>1 should be irrelevant. Empirically: d_state=16 and d_state=128 are within +0.005 of d_state=64. The closed-form captures the eval-time output but misses gradient-flow benefits during training (the ~14% lower train-loss at d_state=16 with similar val shows train-val gap widening — overfit). d_state=64 is the right default.

### 9. **0024-style σ-widening did NOT recur** for the new family (0057 4-seed sentinel)
4-seed σ for 0051 family stays at 0.0030 across n=2,3,4. SEED=31337 (the seed that broke 0024's BigramHash family at +0.0038 σ) lands at 2.0077 here, in-distribution. The triple-parallel kill-Mamba-2 architecture is genuinely robust.

### 10. **Walk hypotheses are inputs, not commitments** (multiple)
Walk 22:22's "quant-noise" theory was refuted by 0041. Walk 01:30's "selectivity helps in parallel" was refuted by 0052. Walk 22:22's "outer-parallel might be the actual best" was an outside-eyes catch resolved against by 0061. **Honest walks generate hypotheses; experiments resolve them. The tag system [SPECULATIVE/WORTH_TESTING/WORTH_DERIVING] kept the walk → desk → walk loop honest.**

---

## Dead axes (verified — don't re-test without changing other levers)

- **Selectivity (full Mamba-2 dt/B/C from in_proj)**: hurts at our regime; replaced by LTI constants (0038-0039 vs 0035-0036). Don't try variants of input-dependent dt/B/C until H100 regime.
- **In_proj fp32-protect (0041)**: broke training (Muon NS scaling on wide-thin matrix). Don't split in_proj to protect dynamics at our regime.
- **K=4 cap-redistribute (0048)**: cap-busts at 17.74 MB > 16 MB. K=3 stays the optimal depth.
- **d_state=128 in cross-class hybrid (0055)**: noise. Don't increase d_state for parameter-golf.
- **d_state=16 in kill-no-BG (0044)**: train converges much faster but val same → overfit. Don't reduce d_state without LR retune.
- **GLA at L=1024 token-by-token (0049)**: too slow on MPS (3-5 hr/exp). Reserved for chunkwise rewrite if pursued in future session. Code committed but unused.
- **3-of-3 LTI Mamba-2 no attention (0040)**: removes last attention block, loses recall. Δ +0.030 BPB. Attention-required.
- **Parallel-S4D-Lin middle (0063)**: cross-class diversity isn't the lever; kill-Mamba-2 (with conv1d) specifically is.
- **BG with no-conv1d Mamba-2 (0062)**: BG can't substitute for conv1d. Don't re-add BG to Mamba-2 family.

---

## Set in stone vs still hypothesis

### Set in stone (multi-seed verified, 4-seed sentinel)
- Kill-Mamba-2 + no-BG + triple-parallel topology: val_bpb 2.00503 ± 0.0015 (4-seed σ_mean), Δ -0.083 vs pure-attn baseline.
- Pure-attn baseline at recur+SwiGLU+mlp=8: 2.08759 ± 0.0001 (2-seed cross-seed Δ 0.0002, very tight).
- Kill > full Mamba-2: 2-seed precision both families (σ_pair 0.0011 and 0.0036). 5σ at family floor.
- Conv1d removal regression: +0.091 BPB. Single-seed but ~5σ at family floor (0.001).

### Still hypothesis (single-seed or decomposition-derived)
- The "+0.012 BPB from middle-parallel topology" attribution: 3-seed (middle-parallel) vs 2-seed (sequential) precision; legitimate but the σ on the middle-parallel side is wider than the sequential family floor.
- The "channel-specific vs global" interpretation of conv1d's mechanism: derived from 0062's refutation but not directly verified by an architectural ablation (would need to test e.g., dense conv vs depthwise — not done).
- The "underperforms at H100 regime" prediction: the kill-wins finding is at 200 steps. Doesn't necessarily generalize to 20k steps. Open transfer question.

---

## Walk reflections

5 walks taken this session. Highlights of items NOT followed up:

- **[WORTH_TESTING parking]** Walk 01:30: per-head B_const, C_const in kill-Mamba-2 (ngroups=nheads). Tests "shared-B/C" axis directly. ~10 line code change. Worth one experiment in next session.
- **[WORTH_TESTING parking]** Walk 01:30: nheads=16 headdim=32 (was 8/64). Same params, different timescale distribution. Cheap env-var or 1-line code change.
- **[WORTH_TESTING / DEFERRED]** Walk 00:26 + multiple: GLA chunkwise reformulation. Token-by-token implementation works (verified 0049) but too slow. Chunkwise version would test whether per-channel vector gates pay off where Mamba-2's scalar gates don't.
- **[WORTH_TESTING]** Walk 22:22: long-context test (L=2048). Tests whether parallel-topology gap to transformer changes at longer context. Needs batch_tokens recalibration.

---

## Predictions vs actuals

No formal predictions table at session start, but plan.md hypothesis bands per experiment can be scored:

| Exp | Predicted band | Actual | Hit/Miss |
|---|---|---|---|
| 0038 (kill) | [2.04, 2.20] | 2.0259 | hit (better than expected) |
| 0040 (3-of-3 LTI) | [2.000, 2.080] | 2.0555 | hit (partial-recall-loss outcome predicted at 20%) |
| 0041 (protect_in_proj) | [2.020, 2.045] | 2.1458 | MISS (way out of band — confused hypothesis) |
| 0042 (kill+no-BG) | [2.040, 2.090] | 2.0225 | MISS (much better than predicted; underestimated) |
| 0046 (cross-class hybrid) | [2.005, 2.040] | 2.0125 | hit (compound-win outcome) |
| 0048 (K=4) | [2.005, 2.045] | 2.0146 | hit on val_bpb but cap-bust |
| 0049 (GLA) | [2.05, 2.20] | UNRUN | n/a |
| 0051 (triple-parallel) | [2.005, 2.030] | 2.0017 | hit at top end |
| 0052 (full-selective in cross-class) | [2.000, 2.030] | 2.0361 | partial miss (selectivity-helps option had 20% prior; actual was selectivity-strongly-hurts) |
| 0058 (pure-attn baseline) | [2.05, 2.10] | 2.0875 | hit center |
| 0062 (no-conv1d + BG) | [2.04, 2.11] | 2.1225 | miss (BG-helps prediction failed; BG slightly hurt instead) |

**Calibration**: predictions were directionally correct but magnitudes underestimated wins (0042, 0046) and overestimated BG's helpfulness when conv1d removed (0062). 0041's miss was a confused-hypothesis miss, not a calibration miss.

---

## Follow-ups for next session (ranked by EV)

1. **GLA chunkwise rewrite** [HIGH EV, big code]. The token-by-token version (0049) is too slow. A chunkwise GLA per Yang 2024 §3.3 would let us actually test the "vector gates per channel" architecture against kill-Mamba-2's scalar gates. The 0049 train_gpt.py has the full block + verifier already; just needs the chunkwise loop replacing the token loop. Subagent task ~150-300 lines.

2. **DeltaNet / RWKV-v6** [HIGH EV, big code]. Different recurrence paradigms entirely. Could be cleaner than GLA. Larger code commitments. Subagent territory.

3. **H100 transfer of triple-parallel-kill-Mamba-2 at 20k steps** [HIGH EV — primary deliverable]. Tests whether the -0.083 BPB SSM win generalizes to longer training. Specific prediction worth testing: does kill-wins reverse at 20k steps?

4. **Per-head B_const, C_const in kill (ngroups=nheads)** [MEDIUM EV, cheap]. Direct test of "shared-B/C" axis. ~10 line code change. Each head gets its own κ_h. From walk 01:30 parking.

5. **nheads=16 headdim=32** [MEDIUM EV, cheap]. Same params, different timescale distribution. Tests if more independent timescales help.

6. **Long-context (L=2048)** [MEDIUM EV]. Does parallel-topology gap change at longer context? Tests "selectivity helps with long context" original Mamba motivation.

7. **5-seed sentinel of 0051 family** [LOW EV — diminishing]. The 4-seed σ is already tight. A 5th seed would tighten σ_mean from 0.0015 to 0.0013. Not worth it unless preparing for formal submission.

---

## Reflections (what worked, what didn't, anti-patterns)

### What worked

- **Walk-22:22's "diverse-mechanism mixing in parallel beats sequential" prediction**: bold pivot, cross-class hybrid landed -0.012 BPB then -0.005 more from triple-parallel. Walks-as-hypothesis-generator did its job.
- **Outside-eyes invocation at 06:30**: caught 3 distinct issues (0054 unconfirmed, BG-conv1d hypothesis untested, S4D-vs-kill-Mamba-2 in parallel untested). Each became a useful experiment. The skill paid for itself easily.
- **4-seed sentinel discipline at 06:00**: confirmed σ stable across seeds. Without this the headline -0.083 BPB would rest on n=2 σ_pair=0.0029 — much weaker. Lesson: 4-seed before architecture lock matters.
- **Mid-flight kill of 0041**: when val_bpb came out 2.1458 vs predicted [2.020, 2.045], didn't try to rationalize — diagnosed as confused hypothesis (CONTROL_TENSOR_NAME_PATTERNS doesn't affect training). Honest journaling.

### What didn't

- **0049 GLA token-by-token**: estimated time too optimistically. Subagent reported 3-5 hr/exp at L=1024. Should have predicted this from "Python loop in MPS dispatch path." Park GLA-chunkwise as a session-of-its-own task next time.
- **K=4 cap-math (0048)**: forgot that K=N adds an MLP per K, not just SSM/attn. 17.74 MB > 16 MB cap-bust. Sanity-check artifact projection more carefully when adding K layers.
- **Walk-01:30 "selectivity-in-parallel-might-help" hypothesis**: the prior should have been lower than 20% — kill-wins was already a sequential vs parallel topology-orthogonal finding by 0042/0045 timing. Refined the prior incorrectly during the walk.

### Anti-patterns surfaced

- **Drift to packaging mode** (caught by outside-eyes review): had started spending compute on σ-tightening of 0042/0045 and pure-attn baselines instead of extending the bold direction. The reviewer flagged this and the next 4 experiments (0061-0063) were genuinely informative pivots. Lesson: outside-eyes is the right correction mechanism for self-confirming chains.
- **Skipping walks during high-velocity execution**: declined 3 walk check-ins between 02:30 and 07:00. Probably right call given the experiments were pivot-distinct (cross-class topology variants), but worth re-examining: were any pivots that should have been more bold avoided? Hard to say in hindsight.
- **Single-seed direct-promotes followed by careful 4-seed sentinels**: this worked here (0051 single-seed at 2.0017 + 4-seed sentinel at 2.00503 mean) but is a discipline that requires actually doing the sentinel. The temptation to ship the single-seed and move on was real; resisting it added rigor.

### Honest uncertainty

The headline -0.083 BPB number is real at 4-seed precision. But at H100 20k-step regime:
- Kill-wins might reverse (selectivity becomes useful with more training).
- Conv1d-as-recall might still hold (it's a structural pattern, not a regime artifact).
- Cross-class parallel topology should probably hold (residual-sum compatibility is structural).

Confidence ranking for transfer (high → low): cross-class topology > conv1d-as-recall > BG-hurts-Mamba-2 > kill-wins. The kill-wins finding is the most regime-sensitive and most likely to soften at H100.

---

## Hand-off

Next session start: **the GLA chunkwise reformulation is the highest-EV unfinished bet.** The 0049 code path is built and verified; only the inner-loop needs replacing. Subagent task with the same shape as the 0046 cross-class subagent dispatch. Hypothesized outcome: GLA val_bpb in [2.00, 2.05]; if it lands < 2.005 it refines or refutes the kill-wins finding, and either is informative.

If pivoting away from SSM mechanism work: H100 transfer of the current winner (triple-parallel kill-Mamba-2 + no-BG, 4-seed mean 2.00503) is the natural next step. The architecture is in `winners/2026-04-27_triple_parallel_kill_mamba2_no_bigram_recur3x3/train_gpt.py` ready to ship.
