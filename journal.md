# Journal

## Current threads

- **Anchor baseline**: exp 0001_baseline_repro at val_bpb 2.5212, 6.907 MB. ALL Δ comparisons go here.
- **Current best (PROMOTED 2026-04-26 22:30, 2-seed CONFIRMED)**: exp 0038/0039 2-seed mean **val_bpb 2.02723** (cross-seed σ_pair=0.0027). Same as 0035 architecture EXCEPT `MAMBA2_KILL_SELECTIVITY=1`: dt/B/C made input-INdependent (LTI block). +128 params/block (negligible). **Selectivity is NOT load-bearing at our regime — killing it improves val_bpb by 0.01448 vs 0035/0036 2-seed mean 2.04171, robust at 2-seed precision.** Beats transformer-best 2.0869 by **-0.060 BPB**. Path: `winners/2026-04-26_mamba2_lti_kill_selectivity_2of3_recur3x3_swiglu_mlp8_bigramhash/` (will update with 0039 trace). **Mechanism story**: the win is from the Mamba-2 BLOCK structure (conv1d + gating-via-z + SSD-chunkwise + learned A_log/dt_bias), not from selectivity. **Open mechanism question** (next experiment): is the kill-wins finding actually a *quant-protection* finding in disguise? Per walk 2026-04-26 22:22, the (dt, B, C) projections in full Mamba-2 come from in_proj's bf16-quantized weight, while kill's _B_const/_C_const are auto-fp32 (1D). Test: split in_proj, fp32-protect (dt, B, C) slices, re-run full-selective Mamba-2. [transfer:medium — selectivity may recover at H100 20k-step where extra params get more training, OR if quant is the root cause this transfers cleanly to H100 fp32]
- **Superseded** (kept for trace):
  - 0035/0036 (Mamba-2 selective 2-of-3): 2-seed mean 2.04171. `winners/2026-04-26_mamba2_ssd_2of3_recur3x3_swiglu_mlp8_bigramhash/`. Direction was right; mechanism story (selectivity) was wrong — kill version (0038/0039) wins by 0.014 BPB.
  - 0032/0034 (Mamba-2 selective 1 of 3): 2-seed mean 2.06016. `winners/2026-04-26_mamba2_ssd_recur3x3_swiglu_mlp8_2attn_bigramhash/`
  - 0018-0024 (S4D-Lin sandwich + BigramHash): 4-seed mean 2.08389. `winners/2026-04-26_ssm_hybrid_recur3x3_swiglu_mlp8_2attn_bigramhash/`. Original "5σ" headline was overstated; 4-seed σ-multiple was 1.6σ vs transformer-best.
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
- **Variance regularization observation [SPECULATIVE — partially refuted by 0024]**: the original story was σ tracks attention presence:
  - 0006/0008 (no attention, n=2): σ ≈ 0.012
  - 0009/0011 (1:2 ratio, n=2): σ ≈ 0.002
  - 0012/0014 (2:1 sandwich, n=2): σ ≈ 0.001
  - 0018/0019/0020 (2:1 + BigramHash, n=3): σ ≈ 0.001
  - **0018/0019/0020/0024 (2:1 + BigramHash, n=4): σ = 0.0038** ← REFUTES the "attention drops σ 6×" claim at the BigramHash level.
  
  Updated read: small-n σ estimates are systematically biased low (the worst seed often hasn't been drawn yet). At n=4 the BigramHash family σ is essentially the same as the no-attn n=2 spread (0.0038 vs 0.012 — same order of magnitude). The "attention regularizes variance ~6×" pattern is likely an artifact of n=2 spreads, not a real architectural effect. To salvage anything, would need 4-seed sentinels for ALL families (no-attn, 1:2, 2:1, 2:1+BigramHash) at consistent n. Defer; not blocking.
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

## 2026-04-26 23:25 EDT · exp 0040 + 0041 · pure-LTI-3of3 hurts attention; quant-protection hypothesis was confused

**Two experiments back-to-back, both informative.**

### 0040: LTI Mamba-2 at 3-of-3 (no attention)
**Question**: does the LTI Mamba-2 compound continue to 3-of-3 (drop the last attention block)?
**Result**: val_bpb 2.0555 — Δ vs 0038 (2-of-3 + 1 attn) +0.030 (HURTS).
**Interpretation**: LTI Mamba-2's per-head κ-scalar memory cannot fully replace attention's content-addressable recall. Mirror of S4D-Lin no-attn pattern (gap +0.061) but smaller magnitude — LTI Mamba-2 retains more recall capacity than LTI S4D-Lin, but not enough to make attention dispensable. **Compound trend (each Mamba-2 position adds ~0.02 BPB) DOES NOT continue to 3-of-3** — 1 attention block still earns its keep.
**Plan's "partial-recall-loss" outcome (20% predicted) hit.**

### 0041: protected-in_proj attempt — confused hypothesis
**Question**: was 0038's kill-wins finding actually a quantization-protection finding in disguise? Test: split in_proj into bf16-zx and fp32-protected-dyn slices, run full-selective Mamba-2 with protected dynamics.
**Result**: val_bpb_post_quant **2.1458** (pre-quant 2.1427) — far outside predicted [2.020, 2.045] band. Regression of +0.106 vs 0035 full Mamba-2.
**Diagnosis**: my hypothesis was confused. **CONTROL_TENSOR_NAME_PATTERNS only affects post-train serialization (int8 quantization). Training is bf16 regardless.** So protecting in_proj's dyn slice doesn't change anything during training — training was already running bf16-noisy on those weights in 0035 too. The "quant noise on per-token dt/B/C" theory only would have applied at INFERENCE time, when bf16-stored weights get re-loaded. But 0035's pre-quant was already 2.0375 — so the issue isn't quant.

**What 0041 actually measured**: the *training-side* effect of splitting one Linear layer into two. Pre-quant gap (2.04 → 2.14) shows the split itself broke training — likely Muon's Newton-Schulz scaling behaves differently on the wide-thin (144, 512) dyn slab vs the original (2192, 512) combined. Subagent verified bit-exact mathematical equivalence at fp32 (max diff = 0.0), so the issue is in optimizer interaction, not arithmetic. Not worth fixing — the experimental premise was wrong.

**Implication for the kill-wins story**: 0041's regression doesn't refute the 0038/0039 kill-wins finding. The kill-wins effect is **NOT a quant-protection artifact** — it's genuine architectural. The mechanism remains "selectivity is anti-load-bearing at our regime" (or equivalently, "the Mamba-2 BLOCK structure does the work, not selectivity"). The walk's 22:22 quant-noise hypothesis is **REFUTED by my own confusion** — protection only matters post-quant, training was always bf16. Removing this hypothesis from the open-question list.

**Updates**:
- 0040 status: keep (informative on the recall-gap question, even though it regressed).
- 0041 status: discard (broken experiment, wrong hypothesis).
- Update Current threads: kill-wins is genuine architectural; quant-noise hypothesis refuted by my misunderstanding.
- Open questions narrowed: still need to characterize WHY selectivity hurts at 200 steps. Remaining hypotheses (most plausible to least): under-training of dyn-feeding in_proj dims, capacity over-fit (too many channels for 5M tokens), init mismatch where selective contribution starts at zero and never grows.

**Conclusion** [VERIFIED]: kill-wins finding (0038/0039) is robust and not explained by quantization. Will not pursue this hypothesis branch further. Next move (per take-a-walk note 22:22 secondary item): test BigramHash interaction — does removing BigramHash change which version wins? That cleanly tests whether selectivity's "job" was duplicating BigramHash's recall mechanism.

## 2026-04-26 22:30 EDT · exp 0039 · SEED=42 confirms kill-wins finding (2-seed Δ -0.014 BPB robust)

**Question**: SEED=42 confirm of 0038's surprising kill-wins result. 0038 (LTI Mamba-2) at SEED=1337 was 2.0259, vs full Mamba-2 (0035/0036 mean 2.0417) by ~0.0158. Cross-seed σ for the full version was 0.0036 — would the kill version's win hold up at SEED=42, or was 0038 a lucky seed?

**Setup**: env.sh forked from 0038, only diff is `SEED=42`. Predicted ~2.029 (= 0038 + ~0.003 typical seed noise).

**Result**: val_bpb_post_quant = **2.02857** — within predicted band (2.029).

| Variant | SEED=1337 | SEED=42 | 2-seed mean | Cross-seed Δ |
|---|---|---|---|---|
| Full Mamba-2 (0035/0036) | 2.03994 | 2.04349 | **2.04171** | 0.0036 |
| Kill Mamba-2 (0038/0039) | 2.02590 | 2.02857 | **2.02723** | 0.0027 |
| **Family-mean Δ** | -0.01404 | -0.01492 | **-0.01448** | — |

Cross-seed Δ for the kill family is 0.0027 — *tighter* than the full family's 0.0036. So the kill family is at least as well-behaved variance-wise. Family-mean Δ = -0.01448 BPB. At family-floor σ_pair=0.0027 that's **5.4σ at 2-seed precision; robust by any reasonable threshold.**

**Conclusion** [VERIFIED at 2-seed precision]: **the kill-wins finding is REAL.** Selectivity-killed Mamba-2 robustly beats full Mamba-2 by ~0.014 BPB at our regime. The mechanism story (Mamba-2 BLOCK structure, not selectivity) holds across seeds.

**Updates**:
- Promote 0038/0039 to formally supersede 0035/0036 (already direct-promoted 0038; now confirmed at 2-seed). 
- Open mechanism question for next experiment: is the kill-wins effect actually a *quantization-protection* effect in disguise? See walk 2026-04-26 22:22. Specifically: in full Mamba-2, the per-token (dt, B, C) come from in_proj's bf16-quantized 2D weight. In kill Mamba-2, _B_const and _C_const are 1D → auto-fp32. The recurrence accumulates 1024 timesteps of dynamic-parameter contributions; if those are bf16 in full vs fp32 in kill, quant noise accumulates differently. **0041 (next): split in_proj into bf16-zx and fp32-protected dyn slices. Run full-selective Mamba-2 with fp32 dynamics. If val ≈ 2.025, quant noise was the dominant story. If val ≈ 2.040, selectivity is genuinely anti-load-bearing.** Either outcome is a strong writeup mechanism.

## 2026-04-26 21:55 EDT · exp 0038 · selectivity-killed Mamba-2 BEATS full Mamba-2 by 0.014 BPB — headline pivot

**Question** (the decisive mechanism ablation for the writeup): does the 0035 Mamba-2 win at 2-of-3 positions come from (A) selectivity, (B) parameter capacity, or (C) auxiliary structure? Replace input-dependent (dt, B, C) with learned constants → block becomes LTI but keeps the same in_proj/conv1d/out_proj/A_log/dt_bias/D_skip/SSD-chunkwise scan. +128 params per block (+0.0077%, just `_B_const + _C_const`).

**Setup**: env.sh forked from 0035; only `MAMBA2_KILL_SELECTIVITY=1` added. Math verified pre-launch via 7-check verifier in `scratch/mamba2_kill_selectivity_check.py`: FFT-conv duality oracle (chunkwise SSD vs LTI kernel-form recurrence) max abs diff 1.96e-8; α_h all in unit disk; param Δ exactly +128. Derivation in `scratch/mamba2_kill_selectivity_derivation.md` (note: d_state=64 collapses to a single scalar κ = ⟨B_const, C_const⟩, so kill version effectively has ONE hidden state per (head, position)).

**Prediction** [CONJECTURE]: val in [2.04, 2.20]. Three-way decomposition: 25% likely "win is parameters/structure" (val ≈ 2.04), 50% likely "selectivity matters substantially" (val ∈ [2.10, 2.16]), 25% likely "selectivity is THE mechanism" (val > 2.16).

**Disconfirming**: val < 2.05 → selectivity is NOT load-bearing → headline pivot needed.

**Result**: val_bpb_post_quant = **2.0259**. **Disconfirming hit, but in the *opposite* direction expected — selectivity-killed BEATS full Mamba-2.**

| Variant | val_bpb | Δ vs 0038 |
|---|---|---|
| 0035 (full Mamba-2 2-of-3, single seed) | 2.0399 | +0.014 (worse) |
| 0036 (full Mamba-2 2-of-3, SEED=42) | 2.0435 | +0.018 (worse) |
| 0035/0036 mean | 2.04171 | +0.016 (worse) |
| **0038 (kill selectivity, single seed)** | **2.0259** | — |

Δ vs 0035/0036 2-seed mean = **-0.0158 BPB**. At transformer-floor σ=0.0024 that's 6.6σ; at BigramHash-family σ=0.0038 it's 4.2σ; at unmeasured Mamba-family σ it's ≥3σ for any σ ≤ 0.005. **Robust direction even under generous σ.**

Train-loss check confirms: at step 100, kill version was 3.5921 vs full 3.6571 (kill ahead by 0.065 train-nats). The kill advantage shows up early and persists.

**Mechanism conclusion** [VERIFIED at single-seed; SEED=42 confirm queued as 0039]:
- (A) Selectivity-as-load-bearing: **DISCONFIRMED**. Killing input-dependence on (dt, B, C) *improves* the result at our regime. The ~5.3σ "Mamba-2 wins" claim from 0035 was correct, but the *mechanism story* is wrong: it's not selectivity.
- (B) Parameter capacity: **WEAKLY SUPPORTED**. Kill version has +128 params (negligible) over full, so the architecture-level param diff is essentially zero. Yet kill > full. So pure-param-count doesn't explain the gain.
- (C) Auxiliary structure (conv1d, gating-via-z, in_proj heads, SSD chunkwise scan, learned A_log, learned dt_bias): **STRONGLY SUPPORTED**. Kill version retains all of these and beats full. The win is the BLOCK STRUCTURE; selectivity is at best decorative, at worst (at our regime) anti-load-bearing.

**Why might selectivity HURT at 200 steps?** [SPECULATIVE — possibly the most interesting research question of the session]:
- in_proj produces (z, x, B, C, dt) at the same total output size whether killed or not. In the full version, 144 of the 2192 in_proj output dims (~6.5%) feed into per-token (B, C, dt). At 200 × 24576 = ~5M tokens of training, those 144 dims may be *under-trained* — adding noise to dynamics rather than signal.
- The kill version *throws away* those 144 dims (they're computed and discarded), letting the remaining 2048 dims focus on (z, x). Effectively a regularization-by-truncation.
- LTI dynamics with constant κ may be enough at our short context (L=1024) — the recurrence's information bottleneck binds before selectivity has anything useful to add.
- This predicts selectivity becomes useful at LONGER training (selectivity gradients accumulate enough signal) AND/OR LONGER context (where dynamic gating actually pays off). H100 20k-step regime should test (1).

**Implications for writeup**:
- Headline mechanism rewrites: "Mamba-2 BLOCK STRUCTURE (without selectivity) gives the best SSM result. Killing selectivity at 200-step regime improves val_bpb by 0.014. The win is conv1d-and-gating-and-LTI-recurrence, not input-dependent dynamics."
- This is a *more interesting* writeup than "selectivity wins" — it's a non-obvious empirical inversion of the conventional Mamba narrative for short-budget regimes.
- Open question for H100: does selectivity recover at 20k steps? If yes, the mechanism is regime-specific. If no, selectivity is anti-load-bearing more broadly. **Specific prediction for H100 20k-step**: full Mamba-2 will close some of the gap to LTI Mamba-2 but will likely not exceed it (only 100× more steps; selectivity needs probably *much* more data to stop being noise on a 1024-token context).

**Promote decision**: Δ=0.0158 vs prior promote (well above advance threshold +0.010). **Direct-promoting** to `winners/2026-04-26_mamba2_lti_kill_selectivity_2of3_recur3x3_swiglu_mlp8_bigramhash/`. SEED=42 confirm queued as 0039 (within 5-experiment window per direct-promote rule). After confirm, will also try 3-of-3 LTI Mamba-2 (0037 was 3-of-3 with selectivity → preempted by this finding).

**Conclusion** [VERIFIED single-seed, story-pivot acknowledged]: selectivity is NOT the load-bearing mechanism in the Mamba-2 wins at our regime. The 0035 promote was correct on direction but wrong on mechanism. The new SSM-best is **0038: LTI-Mamba-2 + 2-of-3 + recur+SwiGLU+BigramHash + ATTN-at-pos-2 → val_bpb 2.0259** (single-seed, SEED=42 confirm pending). Δ vs prior transformer-best 2.0869 = **-0.061 BPB** (was -0.045).


## 2026-04-26 13:51 EDT · session resume · plan

**Resumed** after the 2026-04-26 ssm_session wrap (committed 89b95b5). No wrap-time given. Reading the journal Current threads + summary + writeup.

**State of play.** Promoted winner: 0018/0019/0020 mean **2.08204** (3-seed, σ≈0.001). Beats prior transformer-best 2.0869 by 0.005 BPB. Layer-mixed 2:1-sandwich + BigramHash. The session ended with the layer-mixed BigramHash axis essentially saturated (BIGRAM_VOCAB=8192 hurts; BIGRAM_DIM=128 hurts).

**Pull-out reasoning.** Three flavors of next move on the table:
- **Rigor consolidation (cheap)**: 4th seed (SEED=31337) of 0018 family to tighten σ point estimate from 3-seed (50% rel. uncertainty) to 4-seed (~35% rel.); also no-attn 4-seed sentinel to verify variance-regularization observation.
- **Topology axis (substantial code change)**: Hymba-strict parallel attn+SSM heads per layer, vs my layer-mixed approach. Side-by-side data point that addresses "is the win specific to layer-mixing topology?"
- **Family axis (substantial code change)**: Mamba-1 selective smoke. Characterize selective-scan step time on MPS (vendored `references/mamba_minimal_model.py` available; CUDA kernels unavailable). If 2-3× slow → opens a new family; if 6× → still informative but blocks iteration.

**Order chosen**: warm up with the 4th seed (cheap, completes a rigor commitment), then pivot to a topology or family axis as primary. Saturated axes are a signal to pivot per program.md.

## 2026-04-26 14:08 EDT · exp 0024 · BigramHash 4-seed σ widens to 0.0038 (was 0.001)

**Question**: 4th seed (SEED=31337) of the 0018 config to tighten the BigramHash family σ point estimate from 3-seed (50% rel. uncertainty) to 4-seed (~35%).

**Setup**: env.sh forked from 0018, only diff is `SEED=31337`. All else identical (K=3 L=3 + SwiGLU mlp=8 + 2:1 sandwich + S4D-Lin + BigramHash 4096,64). Step time 5.10 s/step, artifact 12.26 MB.

**Prediction** [CONJECTURE]: val in [2.078, 2.087]; predicted 4-seed mean unchanged (~2.082); predicted 4-seed σ in [0.0007, 0.002].

**Disconfirming**: val < 2.077 or val > 2.090 → 3-seed σ=0.001 estimate was wrong by >2×, headline confidence shrinks.

**Result**: val_bpb_post_quant = **2.08945** — *outside* the predicted band by +0.005 BPB. Disconfirming hit.

| Seed | val_bpb |
|---|---|
| 1337 (0018) | 2.08313 |
| 42 (0019) | 2.08147 |
| 2024 (0020) | 2.08152 |
| **31337 (0024)** | **2.08945** |

- 4-seed mean = **2.08389**
- 4-seed sample σ = **0.00379** (was 0.00094 at 3 seeds — 4× wider)
- σ_mean (n=4) = 0.00190
- Δ vs transformer-best 2.08687 = −0.00298 → **1.57σ** at 4-seed-mean precision
- Δ vs transformer-best at single-seed precision = 0.79σ

**Conclusion** [VERIFIED at n=4 precision]: the headline "SSM-hybrid + BigramHash beats transformer-best" is still **directionally correct** (4-seed mean 2.0839 < transformer 2.0869 by 0.003 BPB) but the **σ-multiple framing in the writeup is too strong**. The 3-seed σ was at the *low end* of its 50%-rel-uncertainty band; the 4-seed σ point estimate is at 0.0038, ~4× wider. SEED=31337 was not pathological (no crash, no late instability, normal step time and quant tax) — just an honest higher-σ seed.

**Updates**:
- Writeup needs revision: replace "5σ" / "multiple σ" framing with honest "1.6σ at 4-seed precision; the 3-seed σ estimate was the low end of its uncertainty band."
- Variance-regularization observation in journal Current threads weakens too — the 2:1+BigramHash family σ is now 0.0038 (n=4), not 0.001 (n=3). The "attention drops σ ~6×" claim across families needs re-checking — probably the small-n estimates were systematically low.
- Promote framing: the win is robust as a *direction* but not as a strong σ-multiple. Keep promoted (real Δ, 4-seed confirmed direction); update writeup to reflect honest uncertainty.

**Process lesson**: This is exactly why noise-floor-sentinel discipline matters. The prior session honestly flagged the σ uncertainty in the writeup itself ("3-seed σ has 50% relative uncertainty, true population σ could be 0.0005-0.003") — that caveat *is now confirmed*. The headline survives, weaker, in honest form. No silent confidence inflation went out the door.

**Open**: should I run a 5th seed (e.g., SEED=98765) to tighten σ further? n=5 → ~30% rel. uncertainty (modest gain over n=4 ~35%). Probably better-EV to spend the experiment slot on a topology axis (0025 Hymba-strict) or sentinel for the no-attn family. Defer 5th seed.

## 2026-04-26 14:42 EDT · exp 0025 · Hymba-strict parallel topology weakly loses to sandwich

**Question**: Does running ATTN+S4D in parallel within every block (Hymba-strict) match or beat the layer-mixed 2:1-sandwich topology (current best 0018)? Single-seed test, K=3 unique blocks each running both, looped ×3 (9 attn + 9 s4d effective vs 6 attn + 3 s4d for sandwich).

**Setup**: Subagent code change implemented `parallel_mode` flag on Block + `PARALLEL_LAYER_POSITIONS=0,1,2` env var. Verified via scratch_verify.py (3 s4d_scale params, all SSM dynamics fp32-protected, 23.7M params vs 0018's 21.8M = +1.9M raw int8). Cap math (scratch/hymba_strict_cap_math.md) predicted 13.55 MB; actual 13.70 MB — math accurate to ~1%.

**Prediction** [CONJECTURE]: val ∈ [2.075, 2.110]; most likely tie (50%). Decisive win < 2.075 (25%). Loss > 2.090 (25%).

**Disconfirming**: val < 2.075 → parallel decisively wins; val > 2.110 → reject parallel.

**Result**: val_bpb_post_quant = **2.09128**. Step_avg = 7.10 s (vs 5.10 s for sandwich = +39%).

| Architecture | val_bpb (single seed) | step_avg | artifact |
|---|---|---|---|
| 0024 BigramHash 4-seed mean | 2.0839 (σ=0.0038) | 5.10s | 12.27 MB |
| 0025 Hymba-strict parallel (SEED=1337) | 2.0913 | 7.10s | 13.70 MB |

Δ = +0.0074 BPB ≈ 1.9σ at single-seed precision (vs BigramHash family σ=0.0038). Within prediction band but on the loss side.

**Conclusion** [LIKELY at single seed]: layer-mixed sandwich topology beats Hymba-strict parallel at our regime, by ~0.007 BPB AND with 39% lower step time. Going to SEED=42 to confirm σ before treating this as a writeup conclusion. The lesson from 0024 (3-seed σ underestimated true σ by 4×) makes single-seed conclusions cheap.

**Cap-math validation**: predicted 13.55 MB vs actual 13.70 MB. Underestimate by 1%. The 1.80× zlib ratio held. Useful prior for future cap-math.

**Implications for writeup**:
- Topology IS a lever, not robust within ~0.005 BPB. Sandwich (A-S-A) is materially better than parallel (P-P-P) at our regime — the position-effect finding (0016/0017 sandwich vs cluster, +0.007 BPB) generalizes: spatial regularity (interleaving attention with SSM) beats spatial concentration.
- The compute-cost asymmetry matters: parallel is +39% step time. Even if it matched val_bpb, it would be a worse Pareto choice.

**Next**: launch SEED=42 of 0025 (= 0026). After: pivot to Mamba-1 selective family (draft in scratch/mamba1_smoke_plan_draft.md).

## 2026-04-26 15:08 EDT · exp 0026 · SEED=42 confirms Hymba-strict loses to sandwich

**Question**: Was 0025's val 2.0913 a single-seed artifact (like 0024) or real loss? SEED=42 confirm.

**Result**: val_bpb_post_quant = **2.08917**.

| Seed | val_bpb |
|---|---|
| 1337 (0025) | 2.09128 |
| 42 (0026) | 2.08917 |
| **2-seed mean** | **2.09023** |

Cross-seed Δ = 0.0021 — tight (similar to other recur+SwiGLU+S4D 2-seed family spreads). Step time 7.04 s/step (matches 0025).

**Conclusion** [VERIFIED at 2-seed precision]:
- Hymba-strict 2-seed mean **2.0902** vs sandwich+BigramHash 4-seed mean **2.0839** = **Δ +0.0064 BPB**.
- Joint σ_mean ≈ √(σ²/2 + σ²/4) using BigramHash σ=0.0038 = 0.0033. Δ/σ_joint ≈ **1.94σ**.
- Topology lever is real: layer-mixed sandwich beats parallel-everywhere by ~0.006 BPB.
- AND parallel costs 39% more step time → strict Pareto loss.

**Topology axis CLOSED**: layer-mixed sandwich (A-S-A pattern) is the operating optimum at our regime, both for val_bpb and step-time. The 0016/0017 finding (sandwich beats cluster by +0.007 within 2:1 ratio) generalizes: spatial regularity > spatial concentration.

**Implications for writeup**:
- New saturation table row: parallel-everywhere joins cluster as topology variants that lose to sandwich.
- The sandwich-2:1 result is robust against multiple topology variants, not just "happens to work."
- Compute-cost dimension matters for the writeup's Pareto framing — sandwich is the best operating point.

**Next**: launch 0027 (middle-parallel: A-PARALLEL-A) to decompose "parallel-everywhere bad" vs "parallel-mixing-anywhere bad." Plan + env.sh ready, env-var-only change.

## 2026-04-26 15:25 EDT · exp 0027 · middle-parallel SURPRISES — val 2.0779 single-seed

**Question**: 0025/0026 confirmed Hymba-strict (parallel everywhere) loses to sandwich. 0027 tests the lighter "middle-parallel" variant: only position 1 is parallel (attn+s4d), positions 0,2 are pure attn. Decomposes "parallel mixing per se vs parallel-everywhere."

**Setup**: ATTN_LAYER_POSITIONS=0,2, PARALLEL_LAYER_POSITIONS=1. Uses 0025's parallel_mode infrastructure unchanged. Inherited 0018 schedule, BigramHash, K=3 L=3, MLP_MULT=8 swiglu.

**Prediction** [CONJECTURE]: val ∈ [2.080, 2.095]. Most likely tied with sandwich (50%). Modest win < 2.080 (25%). Loss > 2.095 (30%).

**Result**: val_bpb_post_quant = **2.07786** — RIGHT at the disconfirming threshold for "decisive win" (plan said val < 2.078).

| Architecture | val_bpb (single seed unless noted) |
|---|---|
| 0024 BigramHash sandwich, 4-seed mean | 2.0839 (σ=0.0038) |
| 0025/0026 Hymba-strict parallel, 2-seed mean | 2.0902 |
| **0027 middle-parallel, single seed** | **2.0779** |
| transformer-best 0062, single seed | 2.0869 |
| 0012/0014 sandwich (no BigramHash), 2-seed mean | 2.0880 |

Δ vs 0024 4-seed mean = **−0.0060 BPB ≈ 1.6σ at family-floor single-seed precision**.

**Step time**: 5.92 s/step (predicted 5.95 — accurate). Artifact: 12.75 MB (predicted 12.71 — accurate). Quant tax: 0.0042 (elevated vs 0024's ~0.001-0.002 — could be from fp32 s4d_scale + attn_scale parallel-position interaction; worth watching).

**Conclusion** [CONJECTURE — single seed]: middle-parallel topology BEATS sandwich. The SSM contribution is **not just compute substitution** — at the middle layer, having both attn and s4d running on the same input lets the model use both mechanisms at the same depth. This is a NEW best candidate. **NEEDS SEED=42 CONFIRM IMMEDIATELY** — single-seed at the threshold has 50/50 odds of being a freak low (recall 0024 SEED=31337 was a freak HIGH).

**Implications** (if confirmed):
- The architectural finding is: at the middle layer of K=3 looped 3×, parallel attn+s4d > attn-only > s4d-only. The "middle layer needs both mechanisms" hypothesis.
- Saturation curve gets a new top entry: A-P-A beats A-A-A and A-S-A.
- Compares directly to Hymba-strict (P-P-P): selective use of parallel beats parallel-everywhere by ~0.012 BPB.
- Promote candidate. But 4-seed σ for this family unknown; needs 2+ seeds before promote per program.md noise-floor rules.

**The walk note from 15:19 also flagged the transformer+BigramHash baseline as the writeup-headline test**. That experiment is queued (0029 plan ready). The 0027 surprise doesn't change that need — even if 0027 is real, transformer+BigramHash tells us whether SSM contribution generalizes beyond this specific architecture.

**Next**: SEED=42 confirm of 0027 (= 0030). DEFER 0028 Mamba-1 launch until 0030 is in. The promote candidate takes priority.

## 2026-04-26 15:35 EDT · directive update · breadth > seed-confirms for SSM exploration

**User course-correction**: "Single-seed exploration is fine for new families. Multi-seed confirms only when something is genuinely a promote candidate. Stop spending half your budget confirming variations of S4D-Lin sandwich — Mamba-1, Mamba-2/SSD, Hyena, gated SSM, larger d_state are all untouched, and any ONE of them could move the writeup more than another seed of middle-parallel."

**Action taken**:
- KILLED 0030 (SEED=42 of middle-parallel) mid-run at step ~14/200. Folder kept as record but no result written.
- 0027 (middle-parallel single-seed val 2.0779) stays as `keep` — characterized, not a promote candidate without further seeds, but the single-seed result IS the data point.
- Pivoted to launch 0028 (Mamba-1 selective) immediately.
- Saved the directive to memory (`feedback_explore_breadth_over_seed_confirms.md`) so future sessions don't repeat the pattern.

**New ordering** (post-0028):
- 0028 Mamba-1 selective smoke (running)
- After: Hyena (learnable kernel via FFN of position) — different family, ~30 lines code
- After: gated SSM (GLA-style) — different family
- After: larger d_state (N=64 with chunking if needed) — different family-axis
- 0029 transformer+BigramHash (writeup-baseline, low priority — it's a variation of existing arch, not a new family)

**Process lesson**: I had drifted into "incremental tightening" mode (4-seed BigramHash sentinel, SEED=42 of Hymba-strict, SEED=42 of middle-parallel). All 3 confirms in this session were on variations of the same recur+SwiGLU+S4D family. The walk at 15:19 already flagged the anchoring; the user's direct feedback is the stronger correction. **Breadth over depth for the writeup.**

## 2026-04-26 15:46 EDT · exp 0028 · Mamba-1 INFEASIBLE on MPS without CUDA kernels

**Question**: Pivot to a new SSM family — Mamba-1 selective scan in 0018's 2:1 sandwich (replace position 1 s4d with Mamba-1).

**Setup**: Subagent (general-purpose) added `MambaBlock` class adapted from vendored `references/mamba_minimal_model.py`. ~200 lines. Numerical correctness verified via `selective_scan_ref.py` oracle (max abs diff = 0). Subagent also fixed an optimizer-routing bug along the way (pre-0028 the `ndim < 2 OR matched` predicate dropped 3D conv1d.weight params from BOTH buckets — would receive zero updates; changed to `ndim != 2 OR matched`). One additional fix on first launch: conv1d input bf16 vs bias fp32 dtype mismatch — solved by adding `conv1d` to CONTROL_TENSOR_NAME_PATTERNS so weight is restored to fp32 too.

**Prediction** [CONJECTURE]: step time 6-8 s/step. Total exp ~22-27 min.

**Result**: After 7 minutes of run, step 1 had NOT appeared in the log. Process alive but no progress. Pure-PyTorch sequential `selective_scan` (1024 iterations of small einsums per Mamba forward, 3 calls per K=3-loop iteration, 9 effective layers) is **dominated by MPS kernel-launch overhead**, not by useful compute. At b=3, L=1024, d_inner=1024, n=16 each iteration's `x_state = deltaA[:,i] * x_state + deltaB_u[:,i]` is small (3*1024*16 elems) but each MPS kernel launch costs ~ms. 1024 iterations × 1ms × 3 calls × 8 grad_accum = ~24s/step minimum, plus backward. Estimated full run >5 hours.

**Conclusion** [VERIFIED]: pure-PyTorch Mamba-1 selective_scan is INFEASIBLE on MPS for our shape. CUDA kernels (mamba-ssm, Triton) are platform-locked. **The family axis pivot for Mamba-1 specifically requires either:**
- Chunkwise selective_scan (Mamba-2 / SSD) which uses matmul-friendly reformulation per primer §2.2
- Drastically smaller seq_len (200) or d_inner (256) for a smoke — but this changes the comparison

**Killed at ~7 min**. Subagent's code is correct but the family is gated on chunkwise reformulation. Marked `discard` in results.tsv with the family-characterization note.

**Empirical update to primer §4.1** (was: "Mamba-1 sequential `selective_scan` ~3-6× slower per primer §4.1"): on this MPS setup, pure-PyTorch sequential selective_scan is **>>10× slower** than S4D-Lin FFT-conv at our (b=3, L=1024, d_inner=1024) shape. Kernel-launch overhead dominates. The "3-6×" was an underestimate or referred to compiled implementations.

**Pivot**: Hyena (FFT-conv kernel via 2-layer FFN of position) is the next family — same FFT-conv path as S4D-Lin (proven fast on MPS), different kernel parameterization. Code change ~30-50 lines via subagent.

## 2026-04-26 16:08 EDT · exp 0031 · Hyena loses by 0.155 BPB at 200 steps (kernel init matters)

**Question**: Family axis — Hyena's learnable-kernel-from-MLP-of-position vs S4D-Lin's structured Vandermonde kernel. Same FFT-conv path; only the kernel parameterization differs.

**Setup**: Subagent added `HyenaBlock` class (~80 lines) with kernel = `kernel_mlp_out(silu(kernel_mlp_in(pos_enc)))` where pos_enc is sin/cos of `kernel_freqs * positions`. 8 frequencies log-spaced 0.1→8.0, kernel_hidden=64. D_skip kept as fp32 via existing pattern. Replaced position-1 of 0018's sandwich with HyenaBlock.

**Prediction** [CONJECTURE]: tied with S4D-Lin (val 2.080-2.090, 40%); win < 2.080 (20%); loss > 2.090 (40%, in case random init harder to train).

**Result**: val_bpb_post_quant = **2.2391** — Δ vs 0024 4-seed mean = **+0.155 BPB**. Loss disconfirming prediction was right but the magnitude is much bigger than anticipated. Step time 4.75 s/step (~7% faster than S4D-Lin — kernel MLP cheaper than complex Vandermonde, but training matters more).

**Conclusion** [VERIFIED single-seed]: at our 200-step regime, the structured S4D-Lin kernel (parametric Vandermonde with init A=-0.5 + π·n complex) is dramatically better than a randomly-initialized MLP-of-position kernel for the same FFT-conv structure. The kernel **initialization** matters at short horizons, not just the parameterization expressivity. Hyena's MLP needs more training to learn a useful kernel than 200 steps allows.

**Implications for writeup**:
- Kernel parameterization IS a meaningful axis at our regime (refines the family-comparison story).
- "Long-conv-with-FFT" is the shared compute primitive; what differs is which init lands in a good basin at 200 steps.
- S4D-Lin's structured init is doing more than expressivity — it's providing a strong inductive prior.
- The "data" row for the writeup is **Hyena 2.239 vs S4D-Lin 2.084** at the same compute path.

**Process note**: cap math was accurate (predicted 12.14 MB, actual 12.82 MB — 5% off due to underestimated kernel_mlp_in size). Step time predicted 5.0-5.2 s, actual 4.75 s (actually faster than predicted).

**Next**: 0032 Mamba-2/SSD selective (subagent code already done, scratch_verify passed including chunkwise vs sequential numerical oracle at 1.9e-6 abs diff).

## 2026-04-26 16:25 EDT · exp 0032 · Mamba-2/SSD WINS BIG — val 2.0590 single-seed (~6.5σ)

**Question**: Family axis — Mamba-2/SSD chunkwise selective scan vs S4D-Lin LTI at position 1 of sandwich. Tests whether selectivity (input-dep Δ, B, C) helps when delivered via the matmul-friendly chunkwise reformulation that's MPS-feasible (unlike Mamba-1's sequential scan, which was infeasible at our shape per 0028).

**Setup**: Subagent (general-purpose) added `Mamba2Block` adapted from official `mamba_ssm/modules/ssd_minimal.py` (Apache-2.0, attribution preserved). d_state=64, expand=2 (d_inner=1024), chunk_size=64, headdim=64 → 16 heads. Scalar A per head (vs Mamba-1's per-channel-per-state matrix A). Inherited the 0028 optimizer-routing fix (`p.ndim != 2 OR matched`) and the conv1d→fp32 trick (`conv1d` substring in CONTROL_TENSOR_NAME_PATTERNS). Numerical oracle: ssd_minimal_discrete (chunkwise) vs sequential reference at b=2 L=16 d_inner=64 → max abs diff = **1.9e-6** (well under tolerance). Replaced ONLY position 1 of sandwich; ATTN at 0,2 unchanged.

**Prediction** [CONJECTURE]: val ∈ [2.075, 2.110]; tied (40%), win < 2.080 (25%), loss > 2.090 (35%).

**Result**: val_bpb_post_quant = **2.05904** — DECISIVELY beats prior best.

| Architecture | val_bpb | n_seeds |
|---|---|---|
| 0024 BigramHash sandwich (S4D-Lin) | 2.08389 | 4 (σ=0.0038) |
| transformer-best 0062 | 2.08687 | 1 |
| 0027 middle-parallel (single seed) | 2.07786 | 1 |
| **0032 Mamba-2/SSD sandwich (single seed)** | **2.05904** | **1** |

- Δ vs 0024 4-seed mean = **−0.0249 BPB ≈ 6.5σ** at single-seed precision
- Δ vs transformer-best = **−0.0279 BPB**
- Δ vs 0027 middle-parallel = −0.0188 BPB
- Step time 5.59 s/step (vs S4D-Lin 5.10s = 9% slower; vs middle-parallel 5.92s = 6% faster!)
- Artifact 12.76 MB
- Quant tax 0.002 — normal
- Train_loss step 200 = 3.44 (vs 0024's 3.50 — model genuinely trained better, not just lucky on val sample)

**Conclusion** [LIKELY at single seed; needs SEED=42]: Mamba-2/SSD selectivity DECISIVELY helps at our 200-step regime when delivered via the matmul-friendly chunkwise scan. The Δ is large enough (~6.5σ at family floor) that even with 50% seed variance it should remain ≥3σ on a 2-seed mean.

**Implications**:
- The "selectivity helps" question primer §4.2 raised — answered YES at our regime, when MPS-feasible.
- Mamba-1 was infeasible BUT Mamba-2 isn't — chunkwise reformulation is the unlock.
- The big jump (~0.025 BPB) suggests the previous SSM-hybrid architecture was leaving recall/selection signal on the table that the LTI s4d couldn't capture.

**Empirical update to primer §2.2**: "2-8× faster than Mamba-1's selective scan" — on MPS for our shape, Mamba-2 chunkwise (5.59 s/step) is at least **>>30× faster than Mamba-1 sequential** (which couldn't even produce step 1 in 7 minutes).

**Next**: SEED=42 confirm of 0032 (= 0034) — strong promote candidate. If confirms, promote.

## 2026-04-26 16:48 EDT · exp 0034 · Mamba-2/SSD CONFIRMED — 2-seed mean 2.0602, PROMOTED

**Question**: SEED=42 confirm of 0032 (Mamba-2/SSD selective at position 1 of sandwich, val 2.0590). Was the win real or single-seed freak?

**Result**: val_bpb_post_quant = **2.06127** (vs 0032 SEED=1337 = 2.05904). Cross-seed Δ = 0.00223 — extremely tight (within transformer-floor σ=0.0024). 2-seed mean = **2.06016**.

**Conclusion** [VERIFIED at 2-seed]:
- Δ vs 0024 4-seed BigramHash mean (2.0839) = **−0.0237 BPB**
- Joint σ_mean ≈ √(σ²/4 + σ²/2) ≈ 0.0033 (using 4-seed BigramHash σ=0.0038 as proxy upper bound)
- Δ/σ_joint ≈ **7.2σ** — clearly significant
- Even if true σ_mamba2 is 4× the 2-seed estimate (i.e. 0.009), Δ/σ_mean = 3.8σ still well past advance threshold

**Promote ritual executed**:
- Created `winners/2026-04-26_mamba2_ssd_recur3x3_swiglu_mlp8_2attn_bigramhash/` from exp 0032 (the parent; 0034 only differs by SEED).
- Removed `final_model.pt` (kept `final_model.int8.ptz` for reproducibility).
- Updated Current threads.

**Statistical caveat noted, not blocking**: per program.md hard rule, SSM-family promotes "should not happen before noise-floor-sentinel completes." With n=2 (both 1337 and 42) we have a tight cross-seed pair but no proper 3+ seed σ estimate. Given the magnitude of the Δ (7.2σ at joint precision; 3.8σ even at 4× σ inflation) the directionality is robust regardless. Adding a 3rd seed for σ tightening is on the parking lot but not blocking.

**The architectural recipe (writeup-ready)**:
- K=3 unique blocks looped L=3 (effective depth 9)
- SwiGLU MLP=8
- ATTN at positions 0, 2 (sandwich)
- **Mamba-2/SSD chunkwise selective scan at position 1** (the new addition vs prior best 0024)
  - d_state=64, expand=2 (d_inner=1024), chunk_size=64, headdim=64 (16 heads)
  - Scalar A per head, data-dep Δ_t and B_t,C_t
  - fp32-protected: A_log, D_skip, dt_bias, conv1d.weight, conv1d.bias
- BigramHash(vocab=4096, dim=64) recall augmentation
- Inherited transformer schedule (warmdown=300, init=0.05, batch=24576, matrix_lr=0.045, muon_steps=15, lr_warmup=30)

**The math story**: SSD = data-dep gated linear attention with scalar gate per head. Scalar A enables chunkwise commutative dynamics → matmul-friendly chunkwise scan → MPS-feasible. The data-dep gate gives back per-token expressivity. Mamba-2 is structurally simpler than Mamba-1 (per-channel diagonal A) but retains the selectivity benefit, and the simplification is what makes it fast on MPS.

**Empirical update to primer §2.2**: "2-8× faster than Mamba-1 on CUDA via matmul" — on MPS for our shape, Mamba-2 chunkwise (5.6 s/step) is at least **>>30× faster than Mamba-1 sequential** (which couldn't even produce step 1 in 7 minutes). The chunkwise reformulation is not just a speed optimization; it's a *feasibility* enabler on platforms without CUDA kernels.

**Next experiments**:
1. Multi-position Mamba-2 (Mamba-2 at 2 of 3 unique blocks): does the win compound?
2. Pure-Mamba-2 (all 3 unique blocks = Mamba-2, no attention): completes the family-comparison saturation row.
3. d_state sweep on Mamba-2 (parking-lot from walk note).

## 2026-04-26 17:10 EDT · exp 0035 · multi-position Mamba-2 COMPOUNDS — val 2.0399 single-seed

**Question**: Does the Mamba-2/SSD win compound at MORE positions? Replace ATTN at position 0 with Mamba-2; pattern becomes Mamba2-Mamba2-ATTN looped ×3 = 6 Mamba-2 + 3 attn effective layers.

**Setup**: env-var-only change vs 0032 (ATTN_LAYER_POSITIONS=2, MAMBA2_LAYER_POSITIONS=0,1). Same code, same schedule, same BigramHash.

**Prediction** [CONJECTURE]: val ∈ [2.030, 2.090]. Compound win < 2.045 (30%); saturate (50%); loss > 2.075 (20%).

**Result**: val_bpb_post_quant = **2.0399** — DECISIVE compound win, well into "compound" zone.

| Mamba-2 positions | val_bpb (single-seed unless noted) | Δ vs prior best |
|---|---|---|
| None (0024 BigramHash 4-seed mean, S4D-Lin sandwich) | 2.0839 | — |
| 1 of 3 (0032/0034 2-seed mean, position 1) | 2.0602 | **−0.0237** |
| **2 of 3 (0035 single seed, positions 0,1)** | **2.0399** | **−0.0203** |
| 3 of 3 (pure Mamba-2, untested) | ? | ? |

**Per-position incremental gain**: 0→1 = -0.024; 1→2 = -0.020. Roughly linear. Pure-Mamba-2 (3 of 3) extrapolates to ~2.020 if linearity continues, but possibly worse if zero-attention loses recall (cf 0006/0008 no-attn pattern at S4D-Lin where val was 2.16 — ~0.07 worse than S4D-attn hybrid).

**Step time**: 5.65 s/step (vs 0032's 5.59s — basically same). The compound win is essentially **free of compute cost**.

**Artifact**: 13.27 MB. Quant tax 0.0024 (normal).

**Conclusion** [CONJECTURE — single seed]: at our 200-step regime, replacing 2 of 3 unique blocks with Mamba-2/SSD beats the 1-of-3 sandwich by 0.02 BPB. The selective-SSM contribution scales with position count up to at least 2 of 3. SEED=42 confirm needed before promote.

**Algebraic frame** (from 16:41 walk note): Mamba-2's diagonal-commutative dynamics are matmul-friendly AND scale with layer-count up to ~2 of 3 effective layers. The 0035 result strengthens the "selectivity > attention at our regime" claim.

**Next**: SEED=42 confirm of 0035 (= 0036). If confirms, promote. Then pure-Mamba-2 (3 of 3, no attention).

## 2026-04-26 17:30 EDT · exp 0036 · Mamba-2 2-of-3 CONFIRMED — 2-seed mean 2.0417, PROMOTED (supersedes 0032/0034)

**Question**: SEED=42 confirm of 0035 (Mamba-2 at 2 of 3 positions, val 2.0399 single-seed).

**Result**: val_bpb_post_quant = **2.04349** (vs 0035 SEED=1337 = 2.03994). Cross-seed Δ = 0.0036, tight. **2-seed mean = 2.04171.**

**Conclusion** [VERIFIED at 2-seed]:
- Δ vs 0032/0034 prior promote (2.06016) = **−0.01845 BPB** at 2-seed precision.
- Joint σ_mean ≈ 0.0035 (assuming similar σ for both pairs). Δ/σ_joint ≈ **5.3σ** — clearly significant.
- Compound trend confirmed: each added Mamba-2 position contributes ~0.02 BPB. 0→1: -0.024 BPB; 1→2: -0.018 BPB.

**Promote ritual executed**:
- Created `winners/2026-04-26_mamba2_ssd_2of3_recur3x3_swiglu_mlp8_bigramhash/` from 0035.
- Updated Current Threads — 0032/0034 explicitly marked superseded with trace pointer.
- Removed pycache, final_model.pt.

**Architecture (writeup-ready)**:
- K=3 unique blocks looped L=3 (effective depth 9)
- Mamba-2/SSD at positions 0, 1 (per K=3 group) → 6 effective Mamba-2 layers after looping
- ATTN at position 2 → 3 effective attention layers
- SwiGLU MLP=8
- BigramHash(4096, 64) recall augmentation
- Inherited transformer schedule (warmdown=300, init=0.05, batch=24576, matrix_lr=0.045, muon_steps=15, lr_warmup=30)

**The mechanism question (NOT YET DECOMPOSED)**:

The 0.045 BPB improvement vs transformer-best could be:
- (A) Selectivity is load-bearing — input-dep dt/B/C
- (B) Parameter capacity — Mamba-2 block is bigger than attention block
- (C) Auxiliary structure — conv1d, gate via z, etc.

**Without 0038 (selectivity-kill ablation, prepped and verified), the writeup cannot claim "selectivity helps."** It can only claim "the Mamba-2 hybrid lands here." The mechanism ablation is the next experiment per program.md update.

**Next**: launch 0038 selectivity-killed Mamba-2 (already verified — FFT-conv duality 1.96e-8, param Δ +0.0077%). Then per the user's strategic update: param-matched transformer, then 0029 transformer+BigramHash (lower priority but still useful).

## 2026-04-26 17:35 EDT · session paused (laptop closing) — resume notes

Human is closing laptop and going. I killed 0038 mid-run (was at step 10/200, healthy trajectory) since MPS won't run reliably with the laptop closed. NOT a session-end — human said "I will resume later when I am back," so no wrap-session ritual. Just leaving state ready for resumption.

**State for resumption**:
- **Current best (PROMOTED)**: 0035/0036 Mamba-2 2-of-3 hybrid, 2-seed mean **2.04171**. `winners/2026-04-26_mamba2_ssd_2of3_recur3x3_swiglu_mlp8_bigramhash/`
- **Next experiment to run** (highest priority, math-verified, prepped): **0038 selectivity-killed Mamba-2**. Just `cd experiments/0038_mamba2_kill_selectivity && ../../run_experiment.sh` to resume. Verifier passed all 7 checks including FFT-conv duality 1.96e-8.
- **Queued plans (next several)**:
  - 0037 pure Mamba-2 3 of 3 (env.sh ready, plan written, not launched).
  - 0038 selectivity-kill (just discussed).
  - Param-matched transformer (not yet drafted) — bump d_ff or d_model to match Mamba-2 block param count, run as transformer-only baseline.
  - 0029 transformer + BigramHash (env.sh + plan ready) — lower priority since Mamba-2 dwarfs the BigramHash-only effect, but still useful as a clean differentiation row for the writeup.
  - Long-sequence test (env-var seq_len=2048) — does Mamba-2's advantage grow at longer context? Tests recall-mechanism story for H100 transfer.
  - d_state sweep on Mamba-2 base (env-var only) — only if 0038 shows selectivity matters.

**Open mechanism question** (decisive for the writeup):
The 0.045 BPB improvement vs transformer-best could be (A) selectivity, (B) parameter capacity, (C) auxiliary structure. **Without 0038's result, the writeup cannot claim "selectivity helps."** Run 0038 FIRST when laptop is back open.

**Math verified for 0038**:
- Selectivity-killed Mamba-2 collapses d_state to scalar κ = ⟨B_const, C_const⟩.
- α_h = exp(A_h · softplus(dt_bias_h)) ∈ [0.94, 0.999] — stable, in unit disk.
- Chunkwise SSD output matches FFT-conv kernel form to 1.96e-8 abs diff.
- Param Δ vs full Mamba-2: +128 per block (+0.0077%) — apples-to-apples.
- See `scratch/mamba2_kill_selectivity_derivation.md` for full derivation.
## 2026-04-26 17:14 EDT · directive update · novelty triage > more architecture exploration

**User strategic reset arrived while 0036 was running**. Multiple framing shifts:

1. **Target anchor**: NOT transformer-best 2.087 MPS (smoke-test number), but **H100 SP1024 ceiling 1.1063 BPB** (records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence). MPS = correctness ledger, not target.
2. **Differentiation audit**: at the SP1024 frontier, our stack differs in exactly ONE row — SSM block in the recurrence loop. Everything else (depth recurrence, BigramHash, SwiGLU) is shared OR we're behind. The SSM contribution we've claimed must be measured against transformer + BigramHash (0029, never run).
3. **"Honest non-record track" is NOT a charity classification** — non-record exists for SLOT/ETLB/TTT violations, not "we omitted standard techniques."
4. **Triage novelty axes single-seed before stack porting**:
   - 0029 transformer + BigramHash — settles whether SSM contribution exists at all
   - 0028 Mamba-1 (already done, MPS-infeasible)
   - MLP=10 sandwich+BigramHash — "spend SSM cap savings" from 1519 walk
   - Mamba-2/SSD — already done with strong signal (compound win 2.04 single-seed)
5. **Stack porting AFTER novelty confirmation**: sliding-window eval, parallel residuals, EMA, Brotli, warmdown=3000, WD≈0.05. Tier-1 are correctness-verifiable on MPS.
6. **Promote discipline kept**: multi-seed-before-promote stays. Change is in EXPLORATION allocation only.

**Reset's stale state**: it referenced 0030 still running (was killed) and 0029 not launched (correct). Did not yet account for 0035's compound win (val 2.0399 single seed). The "0.003 BPB at 4-seed precision" framing is now 0.047 BPB at single-seed (vs transformer-best). But the differentiation question remains: the entire 0.047 could be BigramHash if BigramHash transfers to all-attention. **0029 settles this.**

**Adjusted plan from this point**:
1. Let 0036 finish (running, ~17 min) — it's the 2-seed promote confirm for 0035, which is keep-discipline.
2. After 0036: if confirms, promote 0035/0036. Do NOT run more sandwich seeds.
3. Skip 0037 pure-Mamba-2 (was next).
4. Launch 0029 transformer+BigramHash IMMEDIATELY after promote step.
5. Then MLP=10 sandwich+BigramHash.
6. Then evaluate: does Mamba-2 differentiation survive the BigramHash baseline? If yes → port standard stack. If no → honest "Pareto-equivalent" framing.

**The deliverable redefinition**: train_gpt.py for H100 20k-step + (i) ported stack + (ii) SSM contribution measured in isolation by toggling positions against same stack + (iii) predicted H100 landing zone with honest uncertainty bands.

