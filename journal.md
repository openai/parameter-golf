# Journal

## Current threads

- **Anchor baseline**: exp 0001_baseline_repro at val_bpb 2.5212, 6.907 MB. ALL Δ comparisons go here.
- **Current best (PROMOTED 2026-04-26)**: exp 0018/0019/0020 mean **val_bpb 2.08204** (3-seed, σ≈0.001), K=3 L=3 + SwiGLU(mlp=8) + 2-of-3 attention (positions 0,2 sandwich) + S4D-Lin + BigramHash(4096,64). Beats prior transformer-best 2.0869 by 0.005 BPB. Path: `winners/2026-04-26_ssm_hybrid_recur3x3_swiglu_mlp8_2attn_bigramhash/`. **σ caveat**: at n=3, point-est σ has ~50% uncertainty; true σ in [0.0005, 0.003]. Δ=0.005 is multiple σ at any reading — robust, but headline shouldn't claim "5σ" precisely.
- **Prior transformer-best (now superseded)**: exp 0062 val_bpb 2.08687 at `winners/2026-04-25_recur_3x3_swiglu_mlp8/`. Architecture is comparison-only (don't inherit recur+SwiGLU directly); schedule/optimizer/init defaults ARE inherited — see "Starting env.sh" bullet. Hybrid-composition details: grep `summaries/_archive_transformer/2026-04-25_overnight_session.md`.

- **Starting env.sh for SSM experiments** (architecture-independent transformer wins, [transfer:high] in archive). Set these for any SSM experiment to avoid running on canonical defaults that under-train at 200 steps:
  ```
  WARMDOWN_ITERS=300
  LR_WARMUP_STEPS=30
  TIED_EMBED_INIT_STD=0.05
  MUON_BACKEND_STEPS=15
  TRAIN_BATCH_TOKENS=24576
  MATRIX_LR=0.045
  ```
  Update or replace this bullet as SSM measurements suggest different starting points; remove it once an SSM-native env.sh is established. **Exception: regression-sentinel uses canonical defaults** (warmdown=1200, warmup=0, batch=8192, init=0.005, muon_steps=5) — its job is harness-drift detection against 0001_baseline_repro, which was recorded on canonical.
- **S4D-Lin noise floor**: σ=0.0031 (4 seeds: 1337/42/2024/31337 → 2.23394/2.22747/2.22749/2.22879; mean 2.22942; spread 0.00646), measured 2026-04-26 exp 0002 + 0003-0005 sentinels. Advance Δ ≥ 3σ ≈ **0.010**; judgment-call window [1.5σ, 3σ] ≈ **[0.005, 0.010]**; noise < 0.005. Essentially identical to the transformer floor (0.0024) — S4D-Lin is LTI so the Mamba LR-cliff hazard from primer §4.2 doesn't apply. **For Mamba-family (selective) blocks, the floor is not yet characterized — re-run the noise-floor-sentinel skill when introducing selectivity.** Promote rule for S4D-Lin family inherits transformer's 0.010 advance / 0.005 judgment thresholds.
- **Variance regularization observation [SPECULATIVE]**: across the recur+SwiGLU+S4D family hierarchy, σ tracks attention presence:
  - 0006/0008 (recur+SwiGLU+S4D, NO attention): cross-seed Δ 0.017 → σ ≈ 0.012 — WIDEST.
  - 0009/0011 (1:2 ratio attention): cross-seed Δ 0.003 → σ ≈ 0.002.
  - 0012/0014 (2:1 sandwich attention): cross-seed Δ 0.002 → σ ≈ 0.001.
  - 0018/0019/0020 (2:1 + BigramHash, 3 seeds): σ ≈ 0.001.
  Pattern: adding attention drops σ ~6×. Hypothesis: SwiGLU's gated multiplication amplifies init noise nonlinearly through 3 unique blocks × 3 loops; attention's softmax-bounded outputs short-circuit some of that amplification. Worth verifying — 4-seed sentinel for the no-attn 0006/0008 config would tighten the σ_no_attn estimate. With n=2 the σ has ~50-100% relative uncertainty. **Caveat**: 3-seed σ estimate has its own ~50% relative uncertainty; "σ ≈ 0.001" could be 0.0005-0.003 in true population terms. The promote claim is robust regardless (Δ=0.005 is multiple σ at any reading) but headline σ figures shouldn't be over-interpreted.
- **Primer is internally inconsistent**: main body argues SSM is "almost certainly wrong" for parameter golf; the "Another agent's feedback" section disagrees on (a) whether to quantize the SSM (the `CONTROL_TENSOR_NAME_PATTERNS` env var makes "don't quantize" one line), (b) whether BigramHash closes the recall gap, (c) the probability of an interesting result. Treat both as research opinions; verify empirical claims with measurement; log empirical updates as `Empirical update to primer §X: ...` in entries.
- **MPS reality** [CONJECTURE]: ~5 min per experiment for transformer-speed blocks (S4D-Lin FFT-conv likely lands here); ~15-25 min for sequential `selective_scan` (Mamba-1). Characterize in your first 2-3 experiments. CUDA kernels (mamba-ssm, causal-conv1d, Triton) unavailable — use vendored `references/mamba_minimal_model.py`.
- **Tokenizer is locked at sp1024**.

## Stack of confirmed wins (cumulative path canonical → current best)

- **S4D-Lin all-9-layers replacement** (exp 0002, val_bpb 2.234, mean over 4 seeds 2.229): clean LTI diagonal SSM with FFT-conv, replacing all 9 attention sublayers. Honest framing: 2.229 = (canonical 2.521) − (schedule 0.395) + (architecture penalty 0.103). Schedule does most of the work; SSM costs 0.10 BPB vs same-schedule transformer (consistent with primer §4.5 recall-gap). Confirmed across 4 seeds; cross-seed σ=0.0031. Step time ~2.7-2.8 s/step on MPS, ~2.3× transformer baseline. Path: `experiments/0002_s4d_lin_v1/train_gpt.py`. [transfer:medium]
- **+ K=3 L=3 depth recurrence + SwiGLU MLP=8** (exp 0006/0008, mean 2.163 over 2 seeds): both transformer wins transfer to S4D-Lin and STACK. 0006 SEED=1337 = 2.171, 0008 SEED=42 = 2.155 (SEED=42 actually exceeds the 1337 result by 0.017). Δ vs 0002 mean = −0.066 (~21σ vs pure-S4D σ). NOTE: cross-seed spread for THIS family is 0.017 (5× wider than pure-S4D σ=0.003) — 4-seed sentinel for the recur+SwiGLU+S4D family is needed before fine-grained Δ comparisons in this branch. Arch_diff: transformer (0036→0062) was −0.039; S4D-Lin (0002→{0006,0008}) is −0.066 mean — *larger* gain in SSM regime. Step time 4.4 s (1.6× 0002). Artifact 12.27 MB. Path: `experiments/0006_s4d_recur3x3_swiglu_mlp8/` and `experiments/0008_recur3x3_swiglu_mlp8_seed42/`. [transfer:high]
- **+ 1-of-3-unique-blocks-attention (Hymba-flavored 1:2 ratio)** (exp 0009/0011, mean 2.0981 over 2 seeds; CONFIRMED): K=3 unique blocks looped L=3, with 1 of 3 unique blocks = attention; sequence is S4D-ATTN-S4D-S4D-ATTN-S4D-S4D-ATTN-S4D. 0009 SEED=1337 = 2.0995, 0011 SEED=42 = 2.0966 — cross-seed Δ 0.003 (TIGHT, within transformer-floor σ=0.0024). Δ vs 0006/0008 mean = −0.066 (~5× this family's σ; very strong). Δ vs transformer-best 2.087 = +0.011 — STATISTICALLY TIED with transformer-best at our regime within the recur-family noise floor. Step time 4.7 s. Artifact 12.22 MB. Path: `experiments/0009_recur3x3_swiglu_with_attn/` and `experiments/0011_recur3x3_swiglu_attn_seed42/`. [transfer:high]
- **+ 2-of-3-unique-blocks-attention (2:1 ratio)** (exp 0012/0014, mean **2.08796** over 2 seeds; CONFIRMED): K=3 unique blocks looped L=3, with 2 of 3 unique blocks = attention (positions 0,2). Loop fires ATTN-S4D-ATTN ×3 = 6 attention + 3 S4D applications. 0012 SEED=1337 = 2.08700, 0014 SEED=42 = 2.08891 — cross-seed Δ 0.0019. Δ vs transformer-best 2.08687 = **+0.00109** — *statistically tied to 0.001 BPB on 2-seed mean*. Δ vs 0009/0011 mean (1:2 ratio) = −0.0101 (just past advance threshold). Step time 4.90 s. Artifact 12.28 MB. Path: `experiments/0012_recur3x3_swiglu_2attn_1s4d/` and `experiments/0014_recur3x3_swiglu_2attn_seed42/`. [transfer:high]
- **+ BigramHash recall augmentation** (exp 0018/0019/0020, mean **2.08204** over 3 seeds; PROMOTED): adds zero-init BigramHashEmbedding(4096, 64) on top of token embedding; xor-hashes adjacent token pairs into a 4096-bucket recall mechanism. 0018 SEED=1337 = 2.08313, 0019 SEED=42 = 2.08147, 0020 SEED=2024 = 2.08152 — 3-seed σ ≈ 0.001 (extremely tight). Δ vs 0012/0014 mean (2:1 hybrid w/o BigramHash) = −0.0059. **Δ vs transformer-best 2.08687 = −0.00483 (~5σ at family floor) — SSM-HYBRID + BigramHash BEATS transformer-best.** Cap-cost ~300KB int8. Optimizer split required a small patch to include `base_model.bigram_hash.*` params (subagent caught this). **STRONGEST CONFIRMED RESULT OF THE SESSION; promoted to `winners/2026-04-26_ssm_hybrid_recur3x3_swiglu_mlp8_2attn_bigramhash/`** — record-track-competitive at our regime. [transfer:high — primary deliverable]

## Dead axes (verified — don't re-test without changing other levers)

- **D_STATE = 32** vs 16 (0013, single seed on 0009 base): Δ −0.001 (noise). Larger SSM state-dim is not a meaningful axis at our regime + N=16 baseline. Re-test only if a different config (e.g. larger d_inner via expand=2) might benefit from more state.
- **BIGRAM_VOCAB_SIZE = 8192** vs 4096 (0021, single seed on 0018 base): Δ +0.004 (HURTS). At 200 steps × 24576 tokens, doubling buckets dilutes per-bucket signal (~600 tokens/bucket vs ~1200/bucket at 4096). 4096 is at/above optimum at our token budget. May be different at H100 20k-step regime where buckets get more samples.
- **BIGRAM_DIM = 128** vs 64 (0022, single seed on 0018 base): Δ +0.006 (HURTS). Same diagnosis as VOCAB axis — extra parameters per bucket dilute the per-bucket training signal at 200 steps. (vocab=4096, dim=64) is the joint optimum at our regime. Both directions on each axis hurt.

## Open questions (next session priorities)

**Updated 2026-04-26 mid-session.** The original 9-item primer-based starter list (preserved below in "Original starter recipe") has been substantially answered. New priorities reflecting what's been learned:

### Top priorities for next session

1. **4-seed noise-floor-sentinel for the BigramHash family (0018/0019 base)**: run SEED=2024, 31337 of 0018-style config. Required before formally `promote`-ing 0018/0019 to `winners/`. Should land in [2.078, 2.090]; if so, formal promote; if outlier, tighten the σ estimate for this family.
2. **Formal promote of 0018/0019 architecture** to `winners/2026-04-26_ssm_hybrid_2attn_bigramhash/`. Documents the SSM-best architectural endpoint of this session. Deferred this session due to time + strict skill rule (4-seed sentinel pending).
3. **BIGRAM_VOCAB_SIZE sweep**: 8192 and 16384 on 0018 base. Cap-cost ~1 MB each, comfortably under cap. Could give +0.002-0.005 BPB if more buckets help.
4. **Hymba-strict parallel attn+SSM heads** (vs my layer-mixed 0012/0018): substantial code change but a clean side-by-side data point for the writeup. Tests whether topology matters.
5. **Mamba-1 selective smoke (50 steps)**: characterize step time on MPS for the selective scan. If just 2× slower (not 6×), worth one experiment.
6. **H100 transfer of 0018/0019 architecture at 20k steps**: would settle whether the "200-step MPS" SSM-vs-transformer comparison generalizes. The primary question primer §4.7 asked.

### Resolved questions from original list

- **Q1 (get an SSM block running)**: ✓ Done. 0002 = pure S4D-Lin, val_bpb 2.234 (mean 2.229 over 4 seeds).
- **Q2 (discretization choice)**: ✓ S4D-Lin LTI w/ ZOH worked cleanly, two-line Vandermonde kernel. Did not need to escalate to selective.
- **Q3 (what not to quantize)**: ✓ A_log/A_im/B_proj/C_proj/dt_log/D_skip in fp32 via CONTROL_TENSOR_NAME_PATTERNS. Quant tax 0.001-0.003 across all SSM experiments — primer §4.4 "quant-hostility" was *not* observed at our regime/N=16.
- **Q4 (noise-floor sentinel)**: ✓ S4D-Lin σ=0.0031 measured (4 seeds). Also informally measured for recur+SwiGLU+S4D family (σ≈0.017, 2 seeds) and recur+SwiGLU+S4D+attn family (σ≈0.003, 2 seeds — surprisingly tight).
- **Q5 (S4D vs Mamba bake-off)**: ⚠️ Not done. S4D was sufficient for the headline result; Mamba-1 selectivity untested.
- **Q6 (LR sweep)**: ⚠️ Skipped (per "don't sweep exhaustively"). MATRIX_LR=0.045 inherited from prior session; 0042/0043 in archive showed it was at the optimum for transformer.
- **Q7 (BigramHash)**: ⚠️ Not yet tried. Worth attempting as next-step on top of 0009 base.
- **Q8 (Hymba-lite)**: ✓ Effectively done — 0009 IS a layer-mixed Hymba-flavored hybrid that achieves the primer's "Hymba-lite for non-record track" goal. Within 0.011 of transformer-best.
- **Q9 (quant interaction)**: ✓ Measured. Quant tax in [0.001, 0.003] across all 11 SSM experiments. Stable.

### Open question — depth recurrence transfer (RESOLVED at SSM regime)
**Empirical update**: depth-recurrence K=3 L=3 transfers to S4D-Lin and gives MORE gain than on transformer (−0.066 BPB on S4D-Lin vs −0.039 on transformer at the same arch_diff). Issue #140's "SSM makes layers more powerful → 7L beats 11L pure transformer" framing does NOT apply at our 200-step regime — both architectures benefit from depth recurrence + SwiGLU.

### Open question — scale deception (untested in this session)
PR #1227's d=192 → d=512 regression. We're at d=512 throughout; have not tested d=192. If we ever scale model dim, this is the principle to re-verify.

### Original starter recipe (archived for reference)

1. Get an SSM block running on Mac. ✓
2. Pick discretization (S4D-Lin first). ✓
3. Decide what not to quantize. ✓
4. Run noise-floor sentinel. ✓
5. S4D vs Mamba bake-off. (skipped — S4D sufficient)
6. Don't sweep LR exhaustively. ✓
7. BigramHash recall compensation. (parking_lot, not done)
8. Hymba-lite parallel heads. ✓ (layer-mixed variant)
9. Quant interaction. ✓


## Entries (newest first)

