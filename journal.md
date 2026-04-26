# Journal

## Current threads

- **Anchor baseline**: exp 0001_baseline_repro at val_bpb 2.5212, 6.907 MB. ALL Δ comparisons go here.
- **Inherited transformer best (comparison anchor only)**: exp 0062 val_bpb 2.08687, K=3 L=3 + SwiGLU(mlp=8). Path: `winners/2026-04-25_recur_3x3_swiglu_mlp8/`. Reference for "what an optimized transformer at our regime achieves." **Do not inherit the architecture** (recurrence + SwiGLU MLP=8) — that defeats the SSM exploration goal. **Do inherit the schedule/optimizer/init defaults** below (architecture-independent, [transfer:high]). For full hybrid composition details, grep `summaries/_archive_transformer/2026-04-25_overnight_session.md` for "Recommendations" or "Stack of confirmed wins".

- **Starting env.sh for SSM experiments** (architecture-independent transformer wins, [transfer:high] in archive). Set these in your env.sh for any SSM experiment to avoid running on canonical defaults that under-train at 200 steps:
  ```
  WARMDOWN_ITERS=300
  LR_WARMUP_STEPS=30
  TIED_EMBED_INIT_STD=0.05
  MUON_BACKEND_STEPS=15
  TRAIN_BATCH_TOKENS=24576
  MATRIX_LR=0.045
  ```
  Canonical (warmdown=1200, warmup=0, batch=8192, init=0.005, muon_steps=5) is the pre-fix regime; running an SSM block on canonical confounds architecture signal with under-training. **Exception: regression-sentinel uses canonical defaults** — its job is harness-drift detection against 0001_baseline_repro, which was recorded on canonical.
- **SSM-family noise floor: UNCHARACTERIZED**. The transformer floor of ~0.0024 cross-seed does not auto-transfer. Run the `noise-floor-sentinel` skill on your first stable SSM block. **Until the sentinel completes for an architecture family, do NOT invoke `promote` on any SSM-family experiment in that family — treat any apparent win as informational only.** Mamba's sharp LR cliffs (primer §4.2) make freak-good first-seed runs more likely; the previous transformer session's documented anti-pattern (single-seed direct-promote-zone wins piling up before cross-seed confirms — see `summaries/_archive_transformer/2026-04-25_overnight_session.md` "methodology debt") is the exact failure mode this guardrail prevents. Update this bullet with measured σ and adjusted thresholds when sentinel completes — e.g., "S4D-Lin noise floor: σ=X measured 2026-04-26 exp NNNN-NNNN; advance threshold Δ ≥ 3σ; judgment-call window [2σ, 3σ]."
- **Primer is internally inconsistent**: main body argues SSM is "almost certainly wrong" for parameter golf; the "Another agent's feedback" section disagrees on (a) whether to quantize the SSM (the `CONTROL_TENSOR_NAME_PATTERNS` env var makes "don't quantize" one line), (b) whether BigramHash closes the recall gap, (c) the probability of an interesting result. Treat both as research opinions; verify empirical claims with measurement; log empirical updates as `Empirical update to primer §X: ...` in entries.
- **MPS reality** [CONJECTURE]: ~5 min per experiment for transformer-speed blocks (S4D-Lin FFT-conv likely lands here); ~15-25 min for sequential `selective_scan` (Mamba-1). Characterize in your first 2-3 experiments. CUDA kernels (mamba-ssm, causal-conv1d, Triton) unavailable — use vendored `references/mamba_minimal_model.py`.
- **Tokenizer is locked at sp1024**.

## Stack of confirmed wins (cumulative path canonical → current best)

(empty — populated as SSM wins land. Inherited transformer wins are in `summaries/_archive_transformer/2026-04-25_overnight_session.md`.)

## Dead axes (verified — don't re-test without changing other levers)

(empty — populated as SSM dead axes are verified. Transformer-axis dead-axes from prior session are NOT auto-transferred to SSM regime; verify before assuming.)

## Open questions (next session priorities)

A starting recipe based on primer §4.7 + the primer's "Another agent's feedback" 6-item ranked list. **One researcher's recipe — diverge with reason and document why.** This is a starting menu, not a binding sequence.

1. **Get an SSM block running on Mac iteration loop**. Vendored `references/mamba_minimal_model.py` is the starting point. Goal: forward pass clean, no NaN, get *any* val_bpb. Target: < 2.521 (beats baseline). Primer estimate: half a day.

   **Before training, derive in `scratch/`** (extension of program.md "Measurement over belief" + pull-out's "compute parameter count, sketch the math"). The recurrence amplifies math errors over the sequence length, so untested derivations become smoking guns:
   - **Param count** of the block as a function of `d_inner, d_state, d_conv`. Then *post-quant* artifact size — anything in `CONTROL_TENSOR_NAME_PATTERNS` stays fp32 (4×), so the cap math differs from transformer math (see program.md "SSM-specific harness facts"). Confirm artifact stays under cap.
   - **Eigenvalue placement** of `Ā = exp(ΔA)` post-discretization. For stability you need `|λ| ≤ 1` for all eigenvalues; the standard `A = -exp(A_log)` parameterization gives this for free, but verify with a small print on init.
   - **Kernel formula** for LTI blocks (S4, S4D): `K = (CB̄, CĀB̄, CĀ²B̄, ..., CĀ^(L-1)B̄)`. Compute symbolically on a tiny case and `torch.allclose` against your conv kernel before training.
   - **Numerical agreement against `selective_scan_ref`** for any selective (Mamba-family) scan. Build a small fixed input (e.g. B=2, L=64, D=16, N=8 — adjust to your block); run both your scan and the oracle; `torch.allclose(out, ref, atol=1e-5, rtol=1e-4)` should be True. Debug before training.

   These are the concrete derivations to do for THIS first SSM block. **For HOW to do math well — patterns like worked tiny example, recurrence-vs-convolution as oracle, init invariants, degenerate cases — invoke the `derive-and-verify` skill.** A failed derivation upstream is much cheaper to fix than a "why is this NaN at step 50" debug downstream.
2. **Pick discretization wisely**. Primer suggests S4D-Lin (LTI, ZOH-discretized, two lines of code, debug-friendliest) before Mamba's selective scan. Diverge with reason.
3. **Decide what NOT to quantize**. If the SSM is a small fraction of total params, `CONTROL_TENSOR_NAME_PATTERNS` keeps it fp32 (program.md "SSM-specific harness facts"). The primer's main body and critique disagree on quantization-hostility — measure with vs without protection on your first stable config to settle it for *your* architecture.
4. **Run noise-floor-sentinel** on the working config from (1). ~3 experiments, characterizes architecture-family variance. Required before treating any subsequent Δ as signal.
5. **S4D vs Mamba-1 vs Mamba-2/SSD bake-off** at the same single-replaced-layer position. Determines which selective family to invest in. Each is one experiment with a different vendored block.
6. **Don't sweep LR exhaustively**. Primer suggests 3 points {0.005, 0.01, 0.02} based on Mamba's documented sharp LR cliffs (primer §4.2). Diverge if measurement shows the cliff isn't sharp at your scale.
7. **BigramHash recall compensation** if pure SSM lags on val_bpb (primer §4.5 "Zoology" lesson — 82% of the SSM↔attention perplexity gap is associative recall; BigramHash is record-validated, ~30 lines, subagent task).
8. **Hymba-lite parallel attn+SSM heads** (primer §4.6) is *one* option among several; note that on-leaderboard Hymba (1.1828) and S4D-Lin hybrid (1.1682) both lost to contemporaneous transformers by 0.06–0.10 BPB. Cautious-known-losing has low upside. Less-tried families (GLA, Hyena, RetNet, GateLoop — see PAPERS.md) deserve weight.
9. **Quant interaction**: at first stable SSM config, measure quant_tax. Per primer §4.4, expect amplification; the fp32-protect knob limits damage but may not eliminate it.

### Open question — depth recurrence transfer (untested at SSM regime)
The previous transformer session found a depth-recurrence win (K=3 L=3 looped, +0.0055 vs flat 9L mlp=4 baseline at our 200-step MPS regime). Issue #140 commentary on the Hymba submission stated *"SSM makes each layer more powerful → 7L beats deeper pure transformers at same step budget."* Whether the depth-recurrence instinct transfers to SSM blocks is **untested at our regime**. If you build a hybrid containing both block types, consider testing both directions (looped K=3 vs flat 9L with stronger SSM) rather than assuming Issue #140's framing applies. Log empirical findings as `Empirical update to depth-recurrence question: ...`.

### Open question — scale deception (one documented failure mode)
PR #1227 (SSM hybrid) improved CE 18% at d=192 but regressed BPB 2.7% at d=512. Issue #140 calls this "scale deception" — one PR's failure; whether it generalizes to your architecture is your job to verify. Reasonable practice: if you tune at smaller scale to iterate faster, re-test at the operating scale (d=512) before promotion. Principle transfers; magnitude isn't a law.

## Entries (newest first)
