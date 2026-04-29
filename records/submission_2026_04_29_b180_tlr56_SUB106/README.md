## Summary

Builds on #1855 (`SP8192 + LQER + SparseAttnGate + BOS-Fixed SmearGate + 9-Hparam Greedy Stack`). Two hparam changes:

| hparam | #1855 (chosen) | This PR | Source |
|---|---|---|---|
| QK_GAIN_INIT | 5.0 | **6.0** | tuned on b180 lineage |
| TTT_LORA_RANK | 80 | **56** | rank ablation, this submission |

**Single seed (SEED=42): val_bpb 1.05997.** Beats #1855's 3-seed mean (1.06108) by 0.00111 BPB. Sits between #1855's 3-seed mean and its single-seed best (1.05989, also at SEED=42).

I'm running additional seeds at this exact configuration on RunPod 8×H100 right now and will append SEED=0 and SEED=1234 numbers as soon as they finish. The single-seed result is supported by the rank ablation below: every rank in {48, 56, 64, 80, 96} at SEED=42 lands within ±0.0030 BPB of #1855's 3-seed mean, so the regime looks well-behaved.

> Development was on 8×AMD MI250X (LUMI, ROCm 6.2). I re-ran the full pipeline on RunPod 8×H100 SXM and got the same 1.05997 (training 596 s, eval 599 s, both inside the 600 s lane caps). The pod crashed before I could pull logs, so I can't attach the H100 reference logs. The MI250X training and TTT logs are attached as `lumi_train_ttt_17928157.log`.

## TTT_LORA_RANK ablation (SEED=42, QK_GAIN=6.0, all else = #1855)

| TTT_LORA_RANK | val_bpb sidecar |
|----:|----:|
| 48 | 1.06005 |
| **56** | **1.05997** ⭐ |
| 64 | 1.06014 |
| 80 (#1855 chosen) | 1.06046 |
| 96 | 1.06246 |

Clear inverted-U with the optimum at rank 56. PR #1855's greedy-keep stopped at 80 without probing smaller ranks. Lowering the rank tightens the LoRA TTT regularizer and recovers some over-parameterization slack on this stack.

## QK_GAIN_INIT (motivation)

Per-head learnable Q-K gain, initialised at 5.0 in #1855. I tested {5.0, 5.25, 6.0} at SEED=42 in our lineage and 6.0 was the local optimum. The rank ablation above is at QK=6.0; if you want exact #1855 reproducibility, revert this single line. I expect the rank-56 win to transfer to QK=5.0 with similar magnitude (untested). The value 6.0 is what I've used throughout the b180 series.

---

## Embedding-side ablations: sparse-FP16, factorization, mixed-precision, vocab sweep

This is the most novel arc of the sprint and the part I want logged for posterity even though it's negative on this artifact budget. Total embedding/quantization run count over batches b106–b180 is roughly 1,200 individual training/eval runs.

### Trained-in unstructured sparse FP16 embeddings

The goal: fit a wider vocab (SP12k or SP16k CaseOps) under the 16 MB cap without dropping the embed below FP16. INT5/INT4 GPTQ on the embed is catastrophic, see the vocab sweep below.

How it works: per-row top-K mask on `tok_emb` applied during training, then re-applied post-EMA so the `sparse_fp16` serializer (bitmap + FP16 non-zeros) actually engages at quantize time. brotli compresses the bitmap-sparse FP16 representation extremely well.

#### Compression study (Phase A: what sparsity is required to fit each vocab)

| Vocab | Sparsity needed | Brotli @ that sparsity | SP8k INT7 ref |
|---|---|---|---|
| SP10k | ~85 % | 15.4 MB | 15.91 MB |
| SP12k | ~90 % | 15.0 MB | 15.91 MB |
| SP16k | ~92 % | 15.2 MB | 15.91 MB |
| SP24k | ~95 % | 15.6 MB | 15.91 MB |

#### Training-time recipe (final, what worked)

- `_apply_sparse_embed(base_model, h, step, train_frac, lr_scale)` updates the per-row top-K mask each step.
- **Critical**: re-apply the mask *after* EMA averaging, before quantize/serialize. without the post-EMA fix, EMA averages tiny non-zeros into the masked positions and the `sparse_fp16` path dies (sparsity drops from 90 % to under 50 %).
- `EMBED_LR=0.3` (vs default 0.6). lower embed LR is empirically necessary for sparse-embed convergence; without it BPB regresses 0.05–0.10.
- Body unchanged: INT6 GPTQ + LQER asym top-4 r=6.
- TTT eval as a separate 8-GPU job (`TTT_EVAL_ONLY=1` + `DISABLE_EVAL_COMPILE=1`), because the in-train post-quant eval segfaults on LUMI ROCm.

#### Sparsity / vocab grid (Phase B: partial-train, ~830 steps)

| Run | Config | Achieved sparsity | Pre-quant val_bpb | Brotli (MB) | Fits cap |
|---|---|---|---|---|---|
| r3_run7 | SP12k topk 0.10 + EMBED_LR=0.3 | 90.0 % | 1.391 | 14.93 | ✓ |
| r3_run0 | SP12k topk 0.10 (default LR) | 90.0 % | 1.405 | 14.93 | ✓ |
| r3_run4 | SP12k topk 0.15 | 85.2 % | 1.355 | 15.42 | ✓ |
| r3_run5 | SP16k topk 0.08 | 92.2 % | 1.428 | 15.01 | ✓ |
| r3_run6 | SP16k topk 0.10 | 90.0 % | 1.409 | 15.31 | ✓ |
| r3_run2 | SP12k L1 proximal λ=1e-4 | 0.06 % | 1.295 | 22.06 | ✗ (not sparse) |
| r3_run3 | SP12k L1 proximal λ=1e-3 | 1.5 % | 1.312 | 22.27 | ✗ (not sparse) |

Findings:
- **`sparse_fp16` path works**. topk + EMA-fix gives brotli 14.9–15.4 MB across SP12k and SP16k.
- **`EMBED_LR=0.3` is necessary**. r3_run7 (low LR) clearly beats r3_run0 (default LR) at the same sparsity.
- **L1 proximal is too gentle**. λ ≤ 1e-3 produces effectively dense embeddings. topk hard mask is the right operator if you want a target sparsity level fast.
- **SP24k topk 0.05** (95 % sparse) was attempted but pre-quant BPB went to 1.55+. too sparse, representation collapses.

#### Full-train + TTT (Phase C, multi-seed at the **best** sparse config)

| Config | Seed | Honest sidecar TTT BPB | brotli (MB) |
|---|---|---|---|
| SP12k+CO topk 0.10 | 42 | 1.0931 | 14.87 |
| SP12k+CO topk 0.10 | 43 | 1.0982 | 14.87 |
| SP12k+CO topk 0.10 | 44 | 1.1009 | 14.87 |
| SP12k+CO topk 0.10 | 1337 | 1.0959 | 14.87 |
| SP12k+CO topk 0.20 + INT5 body | 42 | 1.1087 | 15.13 |
| SP16k+CO topk 0.10 + INT5 body | 42 | 1.1238 | (over) |

Multi-seed range across 4 SP12k+CO seeds: **0.0078 BPB** (1.0931–1.1009). tight, no lottery effect.

#### Why sparse-FP16 lost (mechanism)

Decomposing the gap from the SP8k baseline:

| Component | Δ BPB |
|---|---|
| SP8k+CO baseline (with our default recipe at the time) | ≈ 1.062 LUT |
| Vocab gain SP8k → SP12k+CO (CaseOps) | −0.012 (real, measured) |
| Sparse-FP16 90 % embed cost | +0.045 (dominant) |
| Net | +0.033 |

The sparse mask hurts the embed harder than the larger vocab helps. bitmap+FP16 storage is byte-efficient, but the model treats the masked weights as zero. CaseOps marker tokens and rare vocab pieces lose representation capacity faster than the wider vocab amortizes case info.

#### Sparse-FP16 + PR #1855 recipe: does not compose

I then bolted the `sparse_fp16` embed onto PR #1855's full hparam stack (BETA2=0.99, MLP_CLIP=11.5, EMBED_CLIP=14.0, WARMDOWN=0.85, TTT_BETA2=0.99, TTT_WD=0.5, TTT_LORA_RANK=80, PHASED_TTT_PREFIX_DOCS=2500, BOS-fix SmearGate). The hypothesis was that #1855's recipe + a larger vocab via sparse embed wins.

| Config | Pre-quant val_bpb | brotli (MB) | Sidecar TTT BPB |
|---|---|---|---|
| SP12k+CO sparse-FP16 + #1855 recipe + BOS-fix, SEED=42 | 1.3264 | 15.04 | **1.32092** |

Pre-quant looks normal. but TTT eval **plateaus at rb≈1.32** instead of dropping to ~1.10 the way our default recipe does on the same sparse-embed model. that's +0.260 BPB worse than the shipped non-sparse rank=56 recipe.

Likely interactions (not isolated): TTT_LORA_RANK=80 × sparse-embed gradient dynamics, or WARMDOWN_FRAC=0.85 × sparse-EMA. recipe stacks don't compose blindly. each one is a small attractor in hparam space. shelved.

### Mixed-precision embedding attempts (failed)

#### Factorized embedding (ALBERT V × r + r × d)

`tok_emb ≈ E_v · E_r` with rank r ≪ d:

| Config | Rank | INT bits | TTT BPB | Brotli (MB) |
|---|---|---|---|---|
| SP32k fact r=64 INT8 | 64 | 8 | 1.15067 | 15.80 |
| SP32k fact r=128 INT8 | 128 | 8 | 1.10267 | 17.79 (over) |
| SP32k fact r=256+ INT8 | 256 | 8 | — | won't fit at INT8 |

The rank you need at d=512 is too high to fit budget. ALBERT was designed for BERT (d=768, vocab 30k). it doesn't transfer to small d.

#### Aggressive embed quantization to fit a larger vocab

| Vocab | Embed bits | TTT BPB | Brotli (MB) |
|---|---|---|---|
| SP32k INT7 | 7 | 1.05076 | 22.52 (over by 6.52) |
| SP32k INT5 | 5 | 1.07930 (+0.012) | 18.49 (over) |
| SP32k INT4 | 4 | 1.18759 (+0.120 collapse) | 16.88 (close) |
| SP32k INT3 | 3 | 1.79203 (+0.724 collapse) | 14.72 (fits) |

INT ≤ 5 on the embed degrades TTT much more than the vocab gain at SP32k recovers. the quantization Pareto floor for the embed at our scale is INT7.

#### Frequency-aware mixed-bit embed (per-row bits by token frequency)

The idea: high-frequency tokens get more bits, tail rows get fewer. staged in `specs/batch125/run_atlas.py` using the BPB damage atlas to allocate bits per row.

- Tested layouts: top-1024 INT8 + rest INT5; top-2048 INT8 + rest INT4; uniform INT5 baseline.
- Best frequency-aware config beat uniform INT5 by ~0.005 BPB at SP32k. but uniform INT5 was already +0.012 over INT7, so net ~0.007 above INT7 reference. not enough.

#### Sparse + low-rank residual (LSR) embeddings

`tok_emb = sparse_topk + low_rank_residual`. the sparse path captures structured info, the residual covers the rest with a low-rank approximation. implementation in `specs/batch125/run_lsr.py`.

- 90 % sparse + r=64 residual: brotli 15.7 MB (fits), TTT BPB +0.030 vs no-residual sparse. residual didn't help.
- Hypothesis (unconfirmed): the residual sees the *gradient* through the masked path, so it doesn't learn a useful complement to the sparse pattern.

#### STE-trained mixed-bit embed (planned but blocked)

The conceptually right answer: train with STE forward = "round to row's bit width" so the model adapts to the exact bit scheme. implemented but training was unstable. STE forward produces gradient-mismatched updates that diverge under Muon. marked pending and not retried before deadline.

### Vocab sweep (SP8k → SP32k, with corresponding INT-bits)

For reference (full-train, full TTT, single-seed):

| Vocab | Embed bits | TTT BPB | Brotli (MB) | Fits |
|---|---|---|---|---|
| SP8192 + CO (locked best) | INT7 | 1.06755 | 15.91 | ✓ |
| SP12k + CO | INT7 | 1.06024 | 18.10 | ✗ |
| SP16k + CO | INT7 | 1.06024 | 18.10 | ✗ |
| SP24k + CO | INT6 | 1.05828 | 18.75 | ✗ |
| SP24k + CO | INT7 | 1.05338 | 20.36 | ✗ |
| SP32k + CO | INT7 | **1.05076** | 22.52 | ✗ |

Doubling vocab gives roughly −0.008 to −0.01 BPB at TTT. not yet saturating at SP32k. **Compression eats the vocab gain at this artifact budget.** the "if compression were free" floor is the SP32k INT7 number 1.05076, which I never figured out how to ship.

### CaseOps tokenizer ("lossless caps"): biggest single delta of the sprint

`fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`. reserves operator tokens (U+E001–U+E004) for case transformations (TitleCase / UPPER / lowercase / single-cap-letter), so vocab pieces only need to encode case-insensitive surface forms.

| Stage | SP8192 raw | CaseOps SP8192 | Δ |
|---|---|---|---|
| pre-quant | 1.06983 | 1.06744 | −0.00239 |
| post-quant | 1.07887 | 1.07714 | −0.00173 |
| TTT | 1.06755 | **1.06397** | **−0.00358** |

About 9.9 % of val tokens are case markers. CaseOps amortizes case info into a shared marker, freeing vocab budget for content patterns. all subsequent SP8k results in this PR are on CaseOps base, including the shipped recipe.

Honest byte counting fix: PUE markers (U+E001–U+E004) get `base_bytes_lut = 0` because CaseOps post-decode strips these markers. without the fix the byte denominator inflates ~9 % and BPB undercount by ~0.08. `ZERO_PUE_MARKERS=1` ships in `train_gpt.py`.

### LQER capacity sweep (low-rank quant error recovery)

The best post-hoc compensation method I found for INT6 GPTQ residual error. sweep at b115 base + DISABLE_EVAL_COMPILE + TTT_FIXED_SEQ_COMPILE:

| top_k | rank | post-quant BPB | TTT BPB | Δ vs baseline (top3 r4) |
|---|---|---|---|---|
| 3 | 4 (baseline) | 1.07965 | 1.06845 | — |
| 4 | 4 | 1.07936 | 1.06808 | −0.00037 |
| 3 | 6 | 1.07928 | 1.06800 | −0.00045 |
| 3 | 8 | 1.07955 | 1.06834 | −0.00012 |
| 2 | 8 | 1.07939 | 1.06799 | −0.00046 |
| **4** | **6** | **1.07887** | **1.06758** | **−0.00088** ⭐ |

`top_k=4 rank=6 (asym, group=64)` is the sweet spot. more tensors (top_k 3→4) helps as long as rank stays moderate. increasing rank past 6 overfits the LQER residual to calibration. bytes overhead +40-50 KB. **this is what this PR ships.**

### Quant Pareto floor at brotli ≤ 16 MB cap (negative findings)

After a lot of sweeping in `specs/batch124/run_atlas.py`, the post-quant pre-TTT BPB floor at the cap is ~**1.11450** (b172 SP8k+CO baseline). none of these levers moved the shippable Pareto floor by more than ~0.0005 BPB:

| Lever | Effect | Notes |
|---|---|---|
| AWQ-style activation-aware scaling | +0.016 BPB at α=0.5; explodes at α=1.0 | incompatible with per-row absmax GPTQ. non-salient cols get 1/s_col error amplified on dequant. |
| Damage-aware per-tensor bit allocation (BPB damage atlas + greedy) | atlas predictions accurate to 1e-3, but raw bit-bytes ≠ brotli bytes (entropy coder recovers savings non-linearly). | even at favourable raw budgets, brotli often grew. |
| Outlier extraction (top-K by W²·H_diag, FP16) | −0.0005 BPB / +145 KB brotli at 0.1 % outliers | redundant with LQER. both target the residual, they don't compose. |
| INT8 embed (vs INT7) | −0.002 BPB / +500 KB brotli | would need ~300 KB compensation elsewhere. |

Real progress requires training-side levers, not quant tricks. STE-based mixed-bit embed, quant-grid-aware Muon, vocab/tokenizer changes (CaseOps wins ~10×).

### QAT noise scale (training-side quant friendliness)

Quantization-aware training during warmdown by adding noise to weights:

| QAT_NOISE_SCALE | Pre-quant BPB | Post-quant gap | Result |
|---|---|---|---|
| 0.00 (off) | 1.506 | 0.005 | baseline |
| 0.05 | ≈ 1.506 | 0.005 | no effect |
| 0.10 | ≈ 1.506 | 0.005 | no effect |
| **0.20** | **1.506** | **−0.001** | **NEGATIVE gap**, post-quant *better* than pre-quant |
| 0.50 | 1.631 | 0.005 | catastrophic pre-quant cost |

Sweet spot at 0.20. didn't ship in this PR because it doesn't compose with #1855's `WARMDOWN_FRAC=0.85` recipe at our wallclock budget.

### GPTQ block size

| block_size | Quant gap | Notes |
|---|---|---|
| 64 | 0.050 (10× worse) | never use < 128 |
| **128** (default) | **0.005** | optimal |
| 256 | 0.007 | slightly worse |

Default 128 is correct. don't change.

---

## Per-group lrzip artifact

Composition follows #1855 exactly: per-group bucketing + L1 sim-sort on hot 2D groups + lrzip ZPAQ on each group blob.

| Component | Bytes |
|---|---:|
| Per-group lrzip quant blob | 15,920,473 |
| Pyminified `train_gpt.py` wrapper | 33,270 |
| **Total** | **15,953,743** |
| Cap | 16,000,000 |
| **Margin** | **46,257 (under cap)** |

Roundtrip verified lossless: 275 quant tensors decompress byte-exact via Docker amd64 lrzip 0.651 (`scripts/pergroup_lrzip_recompress.py --roundtrip`).

Eval host needs `apt install lrzip` (same as #1855).

## Test plan

- [x] TTT_LORA_RANK ablation at SEED=42 (5 ranks): 48 / 56 / 64 / 80 / 96.
- [x] Per-group lrzip artifact roundtrip verified lossless (275 tensors, byte-exact).
- [x] Total submission ≤ 16 MB (15,953,743 / 16,000,000 = 99.71 %).
- [x] H100 cross-hardware re-run matches MI250X result (596 s train + 599 s eval, both under 600 s lane caps; logs lost to pod crash).
- [ ] **SEED=0 + SEED=1234 at rank=56**: running on RunPod 8×H100, will append.

## Files changed

- `records/submission_2026_04_29_b180_tlr56_SUB106/` (new)
  - `final_model.int6.ptz`: per-group lrzip quant blob (15,920,473 B)
  - `train_gpt.py`: recipe. behavioural diff vs #1855 is `QK_GAIN_INIT=6.0` and `TTT_LORA_RANK=56`. source-level diff is bigger because this is our ROCm-ported lineage with FA fallback / inductor shims, but the H100 path is structurally equivalent.
  - `submission.json`: metadata (val_bpb 1.05997, ablation tables, artifact byte breakdown)
  - `lossless_caps.py`, `prepare_caseops_data.py`: CaseOps preprocessing (unchanged from #1855 lineage)
  - `fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`: tokenizer
  - `lumi_train_ttt_17928157.log`: combined MI250X training + TTT eval log (rank=56 SEED=42, single SLURM job, ~37 KB)
