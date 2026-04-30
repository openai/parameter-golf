# =====================================================================
# READER NOTE — this file is the cumulative experiment-lineage env from
# experiments 0001 → 0124. It is NOT a clean reproduction script. Several
# variables are reassigned multiple times below (e.g. ITERATIONS, MATRIX_LR,
# TRAIN_BATCH_TOKENS, WARMDOWN_ITERS, MAX_WALLCLOCK_SECONDS, VAL_TOKENS,
# NUM_UNIQUE_LAYERS, PARALLEL_LAYER_POSITIONS); under shell semantics the
# LAST assignment wins, and that is what actually ran. Some inline comments
# (e.g. the warnings around MAX_WALLCLOCK_SECONDS=0 and VAL_TOKENS=0) are
# stale — they describe the early MPS-smoke regime, not the final 4×H200
# 1hr deployment.
#
# For the canonical, minimal command that reproduces this submission, see
# the "## Command" section in README.md. That is the source of truth.
# =====================================================================
# Run from the REPO ROOT (where data/ lives), not from this folder.
# train_gpt.py defaults to ./data/datasets/fineweb10B_sp1024 and
# ./data/tokenizers/fineweb_1024_bpe.model — both resolve correctly from
# repo root, matching the convention of the other records-folder submissions.
# (The original ../../data/... paths below were inherited from the
# experiments/<NNNN>/ lineage, which is one directory deeper; they would
# resolve to records/data/... from this folder and break. They are removed
# here so a reviewer who sources this file does not get broken paths.)
export RUN_ID="0124_path_a_h200_1hr"
export VOCAB_SIZE=1024
export ITERATIONS=200
export WARMUP_STEPS=0
# WARMDOWN_ITERS >= ITERATIONS triggers the step-based warmdown from step 0,
# yielding an effective LR ramp of (1200 - step) / 1200 across the whole run
# (0.167 at step 0, ~0 by step 200). This is the regime the Apr-18 reference
# baseline ran in (ITERATIONS=200, WARMDOWN_ITERS unset → defaulted to 1200),
# and it's what our smoke needs: full canonical LR (warmdown_iters << iterations)
# is too aggressive for MPS bf16 numerics and NaNs around step 165.
# To run an experiment at full canonical LR, override per-experiment with a
# small WARMDOWN_ITERS plus an explicit LR_WARMUP_STEPS (10–20).
export WARMDOWN_ITERS=1200
# Wallclock cap disabled so lr_mul uses the step-based warmdown branch (the
# wallclock branch's formula doesn't fire for short smokes — see git log).
export MAX_WALLCLOCK_SECONDS=0
export TRAIN_BATCH_TOKENS=8192
export TRAIN_SEQ_LEN=1024
export VAL_BATCH_SIZE=8192
export VAL_LOSS_EVERY=0
# 16384-token val cap keeps eval ~1 s. Do NOT set to 0 — full val on MPS
# takes 60-120 min per experiment (eval is called twice: pre-quant + post-
# int8-quant). Use SEED=42 re-run for marginal-result confirmation instead.
export VAL_TOKENS=16384
# Dense step logs to catch divergence early; default 200 prints only steps
# 1–10 and step 200, leaving the bulk of training invisible.
export TRAIN_LOG_EVERY=5
# Experiment-specific overrides go below:
# Inherited from 0012:
export WARMDOWN_ITERS=300
export LR_WARMUP_STEPS=30
export TIED_EMBED_INIT_STD=0.05
export MUON_BACKEND_STEPS=15
export TRAIN_BATCH_TOKENS=24576
export MATRIX_LR=0.045
export CONTROL_TENSOR_NAME_PATTERNS="attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,A_log,A_im,B_proj,C_proj,dt_log,D_skip,dt_bias,delta_bias,conv1d"
export NUM_UNIQUE_LAYERS=3
export NUM_LOOPS=3
export MLP_MULT=8
export MLP_TYPE=swiglu
# 0035: 2 of 3 unique blocks are Mamba-2/SSD; 1 attention at end (position 2).
# Pattern per K=3 group: MAMBA2-MAMBA2-ATTN -> looped 3x = 6 mamba2 + 3 attn.
# Tests whether the Mamba-2 win compounds vs the 1-of-3 sandwich (0032/0034).
# 0046: Cross-class hybrid topology. Pattern per K=3 group:
#   pos 0: kill-Mamba-2 (LTI)
#   pos 1: PARALLEL = ATTN || kill-Mamba-2 (sum outputs, scaled)
#   pos 2: kill-Mamba-2 (LTI)
# Tests whether 0027's middle-parallel surprise win compounds with 0038/42's
# kill-Mamba-2 win. Subagent code change: parallel block uses Mamba2Block
# (with kill_selectivity flag) instead of S4DLin in the parallel position.
# 0051: TRIPLE PARALLEL — all 3 unique blocks are PARALLEL (ATTN ||
# kill-Mamba-2). Tests if parallel-topology is universally good or if
# middle-only placement is special. Cap impact: each parallel block ~2.44
# MB; 3 × 2.44 = 7.32 MB SSM+attn (vs 0046's 5.74 MB) → predicted artifact
# ~15.8 MB (cap-tight at 16 MB).
export ATTN_LAYER_POSITIONS=
export MAMBA2_LAYER_POSITIONS=
export PARALLEL_LAYER_POSITIONS=0,1,2
export PARALLEL_SSM_TYPE=mamba2_kill
# 0042: REMOVE BigramHash recall to test if BG was filling selectivity's
# recall niche. If kill still beats full without BG, kill-wins is genuine
# architectural. If full beats kill without BG, BG was making selectivity
# redundant.
export BIGRAM_VOCAB_SIZE=0
export BIGRAM_DIM=64
# 0038: KILL Mamba-2's selectivity. Replace input-dependent (dt, B, C) with
# learned per-head/per-state constants. Same in_proj/conv1d/out_proj/A_log/
# dt_bias/D_skip as full Mamba-2 — only the dynamics become LTI. Decisive
# ablation: if val ≈ 2.04 win is parameters/structure, if val ≈ 2.16 the
# selectivity IS the mechanism. _B_const/_C_const are 1D, auto-fp32.
export MAMBA2_KILL_SELECTIVITY=1
# 0069 COMBINED K=3 + K=4 side-memory. 3-way blend at inference: model +
# K=3 (top_N=100K) + K=4 (top_N=200K) with weights (0.7, 0.10, 0.20).
# Offline analysis (scratch/blend_probe/combined_aggressive_K3.py) predicts:
# blended BPB 1.9504 (delta -0.045 vs model 1.9956 → +0.004 vs K=4-only's
# -0.041). Smoke (_combined_smoke.py) verifies the production pack build
# + 3-way blend matches this within +/- 0.005.
export TRIGRAM_SIDE_MEMORY=1
export TRIGRAM_K=3,4
export TRIGRAM_TOP_N_CTX_K3=100000
export TRIGRAM_TOP_N_CTX_K4=200000
export TRIGRAM_TOP_K=2
export TRIGRAM_BLEND_WEIGHTS=0.7,0.10,0.20
export TRIGRAM_BUILD_TOKENS=100000000
export TRIGRAM_MIN_COUNT=2
# 0074 PER-CONTEXT α: replace the global model weight (0.7 in 0069) with a
# per-context α derived from each trigram context's empirical entropy.
# Low-entropy (confident trigram) → low α (trust trigram more). High-entropy
# → high α (trust model more). Sigmoid mapping with τ, threshold, clip range.
# Adds ~0.2-0.3 MB to the artifact (int8 α buffer per K). Default OFF (=0)
# is byte-identical to parent 0069.
export PER_CONTEXT_ALPHA=1
# Settings tuned via scratch/blend_probe/per_ctx_alpha_sweep.py grid search.
# Best on combined K=3+K=4 offline: BPB 1.9416 vs fixed-α 1.9504, Δ -0.0089.
# Tighter clip range (0.30, 0.85) + lower threshold (3.0) unlocks +0.003 BPB
# more than the subagent's initial defaults (0.5, 0.95, thresh=5.0 → -0.0056).
export ALPHA_TAU=0.5
export ALPHA_THRESH=3.0
export ALPHA_MIN=0.30
export ALPHA_MAX=0.85
# 0076 CONFIDENCE-GATED BLEND: when the model's max log2-prob at a position is
# above this threshold, skip the trigram blend and use model log-probs alone.
# When below, fall back to the per-context α blend (parent 0074 path).
# Best from offline sweep on top of per-context α: -1.0 → BPB 1.9378
# (Δ -0.004 vs per-ctx α alone). Default -1e9 = no gating, byte-identical
# to parent 0074.
export CONF_GATE_THRESHOLD=-1.0

# 0083: ternary body (BitNet b1.58 absmean STE in attn/mamba2/mlp Linears)
export TERNARY_BODY=1

# 0093: combine 0087 LR rescue (MATRIX_LR=0.135) with 0086 v2 packed-ternary base
# Tests if the two confirmed levers stack: LR×3 recovers half ternary penalty, v2 frees cap
export MATRIX_LR=0.135

# 0099: extend training to 2000 steps on CUDA (RTX 5090, ~12x MPS speedup ⇒ ~25 min wall).
# Tests BitNet b1.58 paper claim that ternary penalty closes with more training steps.
# WARMDOWN_ITERS=300 with ITERATIONS=2000 gives [0,1700] constant + [1700,2000] linear warmdown
# in the step-based lr_mul branch (active when MAX_WALLCLOCK_SECONDS=0). Trust ITERATIONS to bound.
export ITERATIONS=2000

# 0102: SSM frontier (kill-Mamba-2 triple-parallel + brotli, val 2.0028 at 0098)
# WITH BitNet b1.58 ternary body + LR×3 — the COMPOUND test of whether the
# 0099 ternary win (transformer + ternary at 2k, pre-quant 1.6499) composes
# with the 0098 SSM win.
#
# Overriding 0099's transformer-with-side-memory topology with the SSM frontier:
export ATTN_LAYER_POSITIONS=
export MAMBA2_LAYER_POSITIONS=
export PARALLEL_LAYER_POSITIONS=0,1,2
export PARALLEL_SSM_TYPE=mamba2_kill
export MAMBA2_KILL_SELECTIVITY=1
export BIGRAM_VOCAB_SIZE=0
# CONTROL_TENSOR_NAME_PATTERNS: SSM frontier set (from 0098), keeps A_log etc fp32:
export CONTROL_TENSOR_NAME_PATTERNS="attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,A_log,A_im,B_proj,C_proj,dt_log,D_skip,dt_bias,delta_bias,conv1d"
# TRIGRAM_SIDE_MEMORY=0 to skip the build cost and avoid the dynamo bug.
# Pre-quant val_bpb is what tests the compound; post-quant requires fixing
# trigram_side_memory.py:634 .item() bug.
export TRIGRAM_SIDE_MEMORY=0
# 2000 steps to match 0099/0100/0101.
export ITERATIONS=2000

# 0102 production-regime override: bump to canonical CUDA batch 524288 (21x larger;
# fits in 5090 32GB easily, fills the GPU). Also REVERT MATRIX_LR to canonical 0.045
# — the 0099 LR×3 (0.135) was tuned at MPS small-batch regime; at canonical batch it
# would be ~3-5x too high. This is the H100-record-recipe applied to the SSM frontier
# with ternary body. Tests whether the compound holds at the deployment regime, not
# just the screening regime. Step time: ~1-2s/step expected at canonical batch on 5090.
export TRAIN_BATCH_TOKENS=524288
export MATRIX_LR=0.045

# Override ITERATIONS down to 1000 since at canonical batch each step sees 21x more
# tokens. 1000 × 524288 = 524M tokens trained, still 10x more than 0099's 49M tokens.
export ITERATIONS=1000

# OOM correction (final override): TRAIN_BATCH_TOKENS=524288 with grad_accum_steps=8
# (hardcoded at world_size=1) gives micro-batch 65536 = 64 seqs of 1024 → ~30 GB
# activations on SSM stack (parallel attn||mamba2 doubles per-block memory). Reduce
# to 131072 (4x smaller, micro-batch 16384 = 16 seqs of 1024, fits in 32 GB easily).
# Still 5.3x larger than MPS-tuned 24576 — gets most of the GPU-fill benefit.
export TRAIN_BATCH_TOKENS=131072

# 0107: bigger SSM + ternary at production batch — tests cap-trade payoff.
# Fork from 0102 (NUM_UNIQUE_LAYERS=3 + ternary + production batch, val 1.5414, 5.6 MB).
# Bump NUM_UNIQUE_LAYERS=5 (5/3 = 1.67× more body params). Predicted artifact:
# 5.6 MB × 1.67 ≈ 9.3 MB, well under 16 MB cap. Question: does the freed cap from
# ternary translate to better val when spent on more depth?
export NUM_UNIQUE_LAYERS=7
export PARALLEL_LAYER_POSITIONS=0,1,2,3,4,5,6

# (n=5 PARALLEL_LAYER_POSITIONS line removed — superseded by n=7 line above)

# 0124 PATH A LONG-TRAIN ROUND 2 on 4×H200 SXM — non-records-track demonstration.
# Goal: scale Path A SSM stack to records-class token budget. Per scratch/2026-04-29_
# record_recipe_analysis.md, "training duration is the dominant gap to records
# (-0.40 to -0.60 BPB)". Records: 6240 steps × 524288 = 3.27B tokens at 1.1063.
# Predicted: 9000-14000 steps × 524288 = 4.7-7.3B tokens (1.5-2.2× records' tokens).
# Predicted post-quant val_bpb [1.15, 1.30].
# Hardware: 4×H200 SXM, $15.96/hr × 3600s = $16.
# Gated on: 0121 (Round 1) running cleanly with predictable step time.
# Peer: non-records "1 Bit Quantization" 1.1239 (Ciprian-Florin Ifrim, 2hr training).
export EMA_BETA=0.999
export EMA_WARMUP_OFFSET=
# 3600s wallclock; ITERATIONS upper bound to avoid step-exhaustion edge case.
export ITERATIONS=20000
export TRAIN_BATCH_TOKENS=524288
# Warmdown ≈ 15% of expected ~12000 steps → 1800.
export WARMDOWN_ITERS=1800
export LR_WARMUP_STEPS=30
# 1hr long-train.
export MAX_WALLCLOCK_SECONDS=3600
# Full eval (writeup-quality).
export VAL_TOKENS=0
