# Journal

**Session protocol**: re-read `scratch/YYYY-MM-DD_session_planning.md` after finishing the first major chunk — context drifts during long sessions; the plan was written when context was fresh, and the chunk-execution may have eroded its framing without you noticing. Drift earned by what you learned is fine; obvious-next-thing drift isn't. Plans are revisable, but only deliberately.

## Current threads

- **Anchor baseline**: exp 0001_baseline_repro at val_bpb 2.5212, 6.907 MB. ALL Δ comparisons go here.
- **Current best (PROMOTED 2026-04-28, 2-seed)**: exp 0076/0077 **2-seed mean val_bpb 1.95141** (cross-seed σ_pair=0.0061). Same architecture as 0069 winner (combined K=3+K=4 static side memory) PLUS: per-context α blend weights (sigmoid of trigram entropy, clip [0.30, 0.85]) + model-confidence gate (skip blend when model max_log2p > -1.0). Path: `winners/2026-04-28_confidence_gated_per_context_alpha_blend/`. Δ vs prior winner (0069 family 1.95990) = **-0.0085 BPB**. Δ vs original 0051 family 2.00503 = **-0.054 BPB**. Δ vs pure-attn baseline (2.088) = **-0.137 BPB**. Step time 8.17 s/step. Artifact 15.91 MB (88 KB safety under 16 MB cap).
- **Cumulative thread-2 contribution**: -0.054 BPB from canonical 0051 baseline (2.005 → 1.951) via 4 stacked mechanisms: brotli + combined K=3+K=4 static side memory + per-context α + confidence gate.
- **Thread 1 closed (2026-04-29)**: AR int6 (0081/0082b) cap-busts in our family — int6-packed bytes near-incompressible by brotli, swap LOSES ~5 MB net despite 25% raw saving. All thread-1 free-score levers now tested. Mini-DR / REPEAT_UNTIE_MLP remain untested but require code changes that conflict with our K=3 L=3 looped triple-parallel topology.
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

**Standing brief**: `scratch/2026-04-28_session_planning.md`. **Thread 1 closed 2026-04-29** — all free-score levers tested. Pivot to thread 2.

Top leads (ordered by EV):

1. **[WORTH_TESTING] Cap-fill: 0065-style asym pos0 + grow K=4 top_N to 280-320K** — only remaining cheap thread-1 win. Env-var only, ~30 min. Predicted -0.004 to -0.005 BPB (offline sweep: `scratch/blend_probe/k4_topn_sweep.py`).
2. **[WORTH_DERIVING] Dendritic N-gram side memory v1 (thread-2 entry)** — warm-start patterns from frequent fineweb 4-grams, train only content vectors. Plan: `scratch/dendritic_memory_plan.md`. Fixes the gradient-sparsity that broke 0073/0080. Subagent ~250 lines.
3. **[WORTH_TESTING] 0071 train-time blend bug fix** — MPS bounds error at B=3 L=1024. Notes: `scratch/0071_train_blend_debug_notes.md`. Tests "model adapts to complement static prior."
4. **[WORTH_DERIVING] H100 transfer of 0076 family at 20k steps** — primary deliverable; specific predictions: kill-selectivity may reverse, conv1d-as-recall + cross-class topology should hold.
5. **[WORTH_TESTING] K=5 static side memory with hash bucketing** — K=5 has 4M+ contexts; needs hashing. Estimated -0.005 BPB.
6. **[SPECULATIVE] Dense-attn HSM (0080) at H100 20k-step** — was neutral at 200 steps; mechanism sound, training duration was bottleneck.
7. **[SPECULATIVE] Bold (e)/(f) from brief** — spike-rank body or dendrocentric layer. Big code, possibly non-record track.

NOTE: pure-learnable on-top-of-static-memory (0073, 0080) NEUTRAL at 200 steps. Don't re-try variants without warm-starting (option d) OR a longer-training plan.

Untested thread-1 levers (NOT free, require code): mini-DR (`RECUR_LAYERS=4,5`, conflicts with K=3 L=3 topology), REPEAT_UNTIE_MLP=full (cap-busts our config; selective version doesn't have a clean mapping in 3-unique × 3-loop scheme).

Parked architecture sub-bets (still on radar, deprioritized): GLA chunkwise rewrite (`experiments/0049_gla_smoke/`); per-head B_const/C_const in kill-Mamba-2; nheads=16 headdim=32; DeltaNet/RWKV-v6; L=2048; conv1d depthwise-vs-dense ablation.

**Next session: dispatch the dendritic memory v1 subagent (lead #2) — that's the canonical thread-2 entry point. If a quick free-score warmup is preferred, lead #1 cap-fill experiment is a 30-minute env-var run.**



## Entries (newest first)

