# Journal

**Session protocol**: re-read `scratch/YYYY-MM-DD_session_planning.md` after finishing the first major chunk — context drifts during long sessions; the plan was written when context was fresh, and the chunk-execution may have eroded its framing without you noticing. Drift earned by what you learned is fine; obvious-next-thing drift isn't. Plans are revisable, but only deliberately.

## Current threads

- **Anchor baseline**: exp 0001_baseline_repro at val_bpb 2.5212, 6.907 MB. ALL Δ comparisons go here.
- **Current best (PROMOTED 2026-04-28, 2-seed)**: exp 0076/0077 **2-seed mean val_bpb 1.95141** (cross-seed σ_pair=0.0061). Same architecture as 0069 winner (combined K=3+K=4 static side memory) PLUS: per-context α blend weights (sigmoid of trigram entropy, clip [0.30, 0.85]) + model-confidence gate (skip blend when model max_log2p > -1.0). Path: `winners/2026-04-28_confidence_gated_per_context_alpha_blend/`. Δ vs prior winner (0069 family 1.95990) = **-0.0085 BPB**. Δ vs original 0051 family 2.00503 = **-0.054 BPB**. Δ vs pure-attn baseline (2.088) = **-0.137 BPB**. Step time 8.17 s/step. Artifact 15.91 MB (88 KB safety under 16 MB cap).
- **Cumulative thread-2 contribution**: -0.054 BPB from canonical 0051 baseline (2.005 → 1.951) via 4 stacked mechanisms: brotli + combined K=3+K=4 static side memory + per-context α + confidence gate.
- **Thread 1 closed (2026-04-29)**: AR int6 (0081/0082b) cap-busts in our family — int6-packed bytes near-incompressible by brotli, swap LOSES ~5 MB net despite 25% raw saving. All thread-1 free-score levers now tested. Mini-DR / REPEAT_UNTIE_MLP remain untested but require code changes that conflict with our K=3 L=3 looped triple-parallel topology.
- **2026-04-29 SESSION HEADLINE — v2 packed-ternary serialization is INFRASTRUCTURE**: BitNet b1.58 ternary body via BitLinear + 2-bit packed export (`pack_ternary` / `unpack_ternary` in `experiments/0086_v2_packed_ternary/modules/bitlinear.py`) frees 56% of artifact cap (16.81 → 7.96 MB) and IMPROVES post-quant val by sidestepping int8 lossy round-trip. Combined with MATRIX_LR=0.135 (×3 LR rescue), best ternary stack lands val 1.993 at 8.21 MB (0093). Still +0.045 BPB worse than 0076 fp/int8 baseline (1.948) but at HALF the cap with ~8 MB headroom. Per BitNet paper, ternary needs ~25× more steps to match fp16; expected to close most of the gap at H100 20k steps.
- **2026-04-29 ternary recipe sensitivity**: BitLinear at default Muon LR=0.045 has +0.10 BPB penalty at 200 MPS steps (0083). LR×3 (0087, 0093) recovers HALF the penalty. Steep recipe slope from a single env var.
- **2026-04-29 split finding on "200 steps too short" hypothesis**: BitLinear body weights ARE LR-bound (recipe-rescuable). Dendritic content vectors (0092, 0094) are NOT LR-bound — training-duration-bound. Same as 0073/0080 — learnable side-content needs more steps regardless of warm-start or LR.
- **2026-04-29 brief's strong-form rank-density claim FALSIFIED at our regime (0095)**: replacing per-(ctx, rank) log2p with global rank template at K=4 lookup REGRESSES val by +0.019 BPB. Per-context calibration carries irreducible information. Caveat: tested rank-only override on same storage, not full permutation-coded R=8 (which would also test storage density).
- **2026-04-29 soft-DP fuzzy K-gram axis closed (0089/0091)**: +40pp coverage gain in offline probe didn't translate to BPB at any FUZZY_DOWNWEIGHT in [0.5, 0.8]. Fuzzy neighbors give noisier predictions than bigram fallback at our regime. Boahen "fringing field" doesn't transfer here.
- **Pure-attn baseline (anchor for writeup)**: 0058/0059 2-seed mean **val_bpb 2.08759** (cross-seed Δ 0.0002). Pure attention 3-of-3 + recur+SwiGLU+mlp=8 + no-BG. Path: `experiments/0058_pure_attn_3of3_baseline/`.
- **Starting env.sh for SSM experiments**: `WARMDOWN_ITERS=300, LR_WARMUP_STEPS=30, TIED_EMBED_INIT_STD=0.05, MUON_BACKEND_STEPS=15, TRAIN_BATCH_TOKENS=24576, MATRIX_LR=0.045`. Schedule defaults are architecture-independent transformer wins; inherit verbatim. Regression-sentinel uses canonical defaults exception.
- **Primer is internally inconsistent**: main body argues SSM is "almost certainly wrong" for parameter golf; the "Another agent's feedback" section disagrees on (a) whether to quantize the SSM, (b) whether BigramHash closes the recall gap. Treat both as research opinions; verify empirically.
- **MPS reality**: ~5 min/exp for transformer-speed blocks; ~8 min/exp for kill-Mamba-2 sequential; ~25 min for triple-parallel. Mamba-1 sequential scan untested — out of scope. CUDA kernels unavailable.
- **Tokenizer locked at sp1024**.

## Confirmed-paying axes (durable knowledge, don't re-derive)

- **Schedule + recur+SwiGLU+mlp=8** (architecture-independent, transfers across SSM families): -0.395 BPB on canonical → 2.087.
- **Mamba-2 BLOCK > S4D-Lin BLOCK** at our regime: -0.044 BPB. Conv1d is the load-bearing differentiator (verified by 0047 ablation: removing conv1d regresses +0.091 BPB).
- **Kill-selectivity > full-selectivity at 200-step regime**: -0.014 BPB. LTI dt/B/C constants beat input-dependent projections because the latter are under-trained at 5M tokens. Verified at 2-seed (0038/0039 vs 0035/0036).
- **No-BigramHash > BigramHash for Mamba-2 family**: -0.005 BPB. BG is redundant with conv1d's recall function in Mamba-2 (opposite of S4D-Lin where BG helps +0.011). BG cannot substitute for conv1d when conv1d is removed (0062 refuted that).
- **Cross-class parallel topology > sequential composition**: -0.012 BPB at middle-parallel, additional -0.005 at triple-parallel. SPECIFIC to ATTN || kill-Mamba-2 pairing (parallel-S4D-Lin loses by +0.021 — 0063). The pairing requires conv1d in the SSM partner.
- **Cross-seed σ for Mamba-2-derived families**: kill+BG σ_pair=0.0036 (n=2); kill+no-BG σ_pair=0.0011 (n=2); middle-parallel σ=0.0027 (n=3); triple-parallel σ=0.0030 (n=4). Tighter than 0024 BigramHash family's 0.0038.

## Dead axes (verified — don't re-test without changing other levers)

- **D_STATE = 32 / 16 / 128** vs 64 (0013, 0044, 0055): all within noise of d_state=64. κ-scalar collapse derivation predicts this; d_state=64 is right default.
- **BIGRAM_VOCAB_SIZE = 8192** vs 4096 (0021): Δ +0.004 (HURTS).
- **BIGRAM_DIM = 128** vs 64 (0022): Δ +0.006 (HURTS).
- **BigramHash on Mamba-2 family** (0042/0043, 0062): hurts +0.005 to +0.009 BPB. Conv1d already does its job.
- **Selectivity (full Mamba-2 dt/B/C from in_proj)**: hurts at 200-step regime; LTI is better. 0035/0036 vs 0038/0039 (2-seed). Don't try variants without testing at H100 regime.
- **In_proj fp32-protect (0041)**: broke training (Muon NS scaling on wide-thin matrix). Don't split in_proj.
- **K=4 cap-redistribute (0048)**: cap-busts at 17.74 MB > 16 MB (each K adds an MLP). K=3 is right depth.
- **3-of-3 LTI Mamba-2 no-attention (0040)**: removes last attention block, +0.030 BPB regression. Attention required.
- **Parallel-S4D-Lin in middle (0063)**: cross-class diversity isn't generic; needs kill-Mamba-2 specifically.
- **Hymba-strict topology with full-Mamba-2+BG (0025/0026)**: lost; but kill+no-BG triple-parallel wins (0051). Topology + base architecture are interactive; don't re-test parallel-everywhere with full-Mamba-2.
- **NUM_LAYERS=11** (0019 archive): +0.0025 noise. Depth ceiling at 200 steps.

## Open questions (next session priorities)

**Standing brief**: `scratch/2026-04-29_session_planning.md` — single-thread, exploratory: SNN / temporal-rank / 1-bit-per-param at LM scale. Read it after `program.md`.

Free-score / port work from prior sessions is closed. The session is research, not lever-pulling. The brief is intentionally a question + resources, not a candidate menu — the direction is yours to invent after derivation.

State of prior thread-2 attempts (so the work isn't repeated unwittingly):

- **Static N-gram side memory (0067-0077)**: -0.054 BPB cumulative, promoted as 0076/0077, but tagged `[transfer:medium — gain regime-specific]`. Not what the SNN/temporal-rank thread is asking about.
- **0073 hash-HSM and 0080 dense-attn HSM**: both NEUTRAL at 200 steps. Prior interpretation was "200 steps too short for any learnable on-top." Re-verify if you want to build on this — the conclusion may be load-bearing for directions you take.
- **0081 (AR self-gen GPTQ int6)**: smoke-OK, NOT launched. It's a thread-1 lever; launching it would re-enter port-mode rhythm. Leave it on the shelf.
- **Prior-session derivations** (UU#1 temporal-rank capacity sim, UU#2 spike-mapping analysis, dendritic-memory design sketches): archived in `scratch/_archive_prior_sessions/`. They biased the previous session toward the conservative warm-start interpretation. Re-derive in your own context if you want to build on them; don't trust the conclusions because someone wrote them down.

Parked architecture sub-bets (independent of the thread-2 frame, still on radar): GLA chunkwise rewrite (`experiments/0049_gla_smoke/`); per-head B_const/C_const in kill-Mamba-2; nheads=16 headdim=32; DeltaNet/RWKV-v6; L=2048; conv1d depthwise-vs-dense ablation. None of these are the thread-2 question.

Untested levers from 2026-04-29 session (worth picking up):
- **H100 cascade** (highest priority — see `summaries/2026-04-29_bitnet-ternary-v2-packed.md` Tier 1 + `scratch/2026-04-29_h100_experiment_design.md`): 3 experiments × ~10-15 min each on 8×H100. Exp 1 = 0093 stack at 20k steps with `TRIGRAM_SIDE_MEMORY=0` (clean ternary baseline); Exp 2 = + dendritic v1; Exp 3 = + static side memory (full stack vs 0076 H100). H100 hyperparameters need re-tuning (batch, warmdown) per H100 records.
- **MATRIX_LR ×5 or ×10 on ternary body** (0093 stopped at ×3, recovered half penalty; further LR could close more). Cheap MPS test.
- **Brief option (f) dendrocentric layer (replace MLP with dendrite bank)**: never built. Subagent task ~250 lines. The most-aligned-with-Boahen swing.
- **Brief option (c) spike-rank embedding**: never built.
- **Brief option (e) full spike-rank body**: never built. Hardest swing.
- **Rank-coded with FULL permutation storage (R=8 token indices, not template override)**: 0095 tested only the decode-semantics form (Option 3). Full storage-density form needs subagent build.
- **0084 long-kernel conv1d at H100 20k**: regressed at MPS 200 (kernel=4 saturated) but might help at H100 with more training.

**Next session FIRST ACTION**: launch H100 Exp 1 (0093 stack ternary-only at 20k steps) per `scratch/2026-04-29_h100_experiment_design.md`. Need to retune H100 hyperparams from `records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence/`. Tells us in ~10-15 min whether ternary scales as BitNet predicts.




## Entries (newest first)

