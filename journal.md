# Journal

**Session protocol**: re-read `scratch/YYYY-MM-DD_session_planning.md` after finishing the first major chunk — context drifts during long sessions; the plan was written when context was fresh, and the chunk-execution may have eroded its framing without you noticing. Drift earned by what you learned is fine; obvious-next-thing drift isn't. Plans are revisable, but only deliberately.

## Current threads

- **Anchor baseline**: exp 0001_baseline_repro at val_bpb 2.5212, 6.907 MB. ALL Δ comparisons go here.
- **Current best (PROMOTED 2026-04-27 06:00, 4-seed SENTINEL)**: exp 0051/0053/0056/0057 **4-seed mean val_bpb 2.00503** (sample σ=0.0030, σ_mean=0.0015). Architecture: K=3 L=3 + SwiGLU(mlp=8) + every K=3 unique block is PARALLEL ATTN || kill-Mamba-2(LTI) + NO BigramHash. Path: `winners/2026-04-27_triple_parallel_kill_mamba2_no_bigram_recur3x3/`. Δ vs pure-attn baseline (0058/0059, 2-seed mean 2.08759) = **-0.0826 BPB**. Step time 8.17 s/step. Artifact 15.18 MB (cap-tight). Family σ stable across n=2,3,4. [transfer:high]
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

1. **GLA chunkwise rewrite** [HIGH EV, big code]. Code exists in `experiments/0049_gla_smoke/train_gpt.py` (token-by-token, verified by 5/5 numerical checks but too slow at L=1024). Replacing inner loop with paper §3.3 chunkwise scan would let us actually test "vector gates per channel" vs Mamba-2's scalar gates. Subagent task ~150-300 lines.

2. **H100 transfer of triple-parallel-kill-Mamba-2 at 20k steps** [HIGH EV — primary deliverable]. Architecture in `winners/2026-04-27_triple_parallel_kill_mamba2_no_bigram_recur3x3/train_gpt.py`. Specific prediction: kill-wins (selectivity-anti-load-bearing) might reverse at 20k steps; conv1d-as-recall and cross-class-topology should hold.

3. **Per-head B_const, C_const in kill-Mamba-2 (ngroups=nheads)** [MEDIUM EV, cheap]. Direct test of "shared-B/C-across-heads" — separate from selectivity. ~10 line code change. Each head gets its own κ_h. Would refine the κ-scalar collapse story.

4. **nheads=16 headdim=32** [MEDIUM EV, cheap env-var or 1-line]. Same params, different timescale distribution.

5. **DeltaNet / RWKV-v6** [HIGH EV, big code]. Different recurrence paradigms. Larger commitments. Subagent territory.

6. **Long-context (L=2048)** [MEDIUM EV]. Tests "selectivity helps with long context" claim from original Mamba — does the parallel-topology gap change?

7. **Conv1d mechanism refinement** [MEDIUM EV, code]: depthwise vs dense conv ablation; tests whether channel-specificity is what makes conv1d load-bearing. Walk 22:22 speculation.

**Next session: GLA chunkwise rewrite is the highest-EV unfinished bet. The 0049 train_gpt.py has the block + verifier; only the inner loop needs replacing per Yang 2024 §3.3.**

## Entries (newest first)

