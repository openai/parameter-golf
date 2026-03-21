# Parameter Golf: Track 10min/16MB Submission Analysis

## Summary Leaderboard (sorted by val_bpb, best first)

| Rank | Folder | Name | Author | val_bpb | val_loss | Quant | Compression | Artifact Size | Steps | ms/step |
|------|--------|------|--------|---------|----------|-------|-------------|---------------|-------|---------|
| 1 | `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` | 10L Int5-MLP + BigramHash(10240) + SWA | thwu1 | **1.14276** | ~1.930 | int5/int6 mixed | zstd-22 | 15,830,186 | 6,711 | 89.42 |
| 2 | `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` | Int6 MLP3x + SmearGate + BigramHash + SWA | Raahil Shah | **1.14582** | 1.93466 | int6 | zstd-22 | 15,862,650 | 7,379 | 81.3 |
| 3 | `2026-03-19_MLP3x_QAT_Int6_SlidingWindow` | 11L MLP3x + Int6 QAT + zstd-22 + Sliding | aruniyer | **1.15015** | 1.94198 | int6 QAT | zstd-22 | 15,427,455 | 10,070 | 59.6 |
| 4 | `2026-03-19_smeargate_orthoinit_muonwd` | SmearGate + OrthoInit + Muon WD | (unnir) | **1.1556** | 1.9511 | int6 QAT | zstd-22 | 15,878,809 | 12,047 | 49.80 |
| 5 | `2026-03-19_WarmdownQuantization` | Int6 MLP3x Sliding Window | samuellarson | **1.15744** | 1.95429 | int6 PTQ | zlib-9 | 15,977,717 | 7,199 | 83.35 |
| 6 | `2026-03-19_Seq2048_FP16Emb_TunedLR` | 10L Int6 QAT + Zstd MLP2.6x | yahya010 | **1.15862** | 1.95628 | int6 QAT | zstd-22 | 15,558,319 | 8,319 | 72.13 |
| 7 | `2026-03-19_int6_STE QAT_ MLP_bigram _U_Net` | *(no submission.json -- OVER 16MB)* | unknown | *1.15978* | 1.95824 | int6 QAT | zstd-22 | **16,186,500** | 12,123 | 49.49 |
| 8 | `2026-03-19_MixedQuant_Int6Int8_SlidingWindow` | Mixed Quant int6/int8 + Sliding Window | aquariouseworkman | **1.16301** | 1.96370 | int6+int8 mixed | zlib-9 | 15,353,490 | 12,395 | 48.41 |
| 9 | `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` | SW + FP16 Emb + 10L + Muon WD + OvertoneInit | notapplica | **1.17475** | 1.98352 | int8 | zlib | 15,374,243 | ~10,500 | ~57.0 |
| 10 | `2026-03-17_LoRA_TTT` | LoRA TTT | sam (samacqua) | **1.1929** | 2.0142 | int8 | zlib | 15,882,446 | ~13,780 | ~43.5 |
| 11 | `2026-03-19_SlidingWindowEval` | Sliding Window Eval (stride=64) | Matthew Li | **1.19250** | 2.01348 | int8 | zlib | 15,874,829 | 13,450 | 44.61 |
| 12 | `2026-03-19_TrainingOptSeq4096` | Training Opt Seq4096 v1 | Spokane Way | **1.20143** | 2.02857 | int8 | zlib | 15,868,326 | 8,394 | 71.47 |
| 13 | `2026-03-18_LongContextSeq2048` | Long Context Seq2048 v2 | Spokane Way | **1.20576** | 2.03588 | int8 | zlib | 15,867,270 | 11,564 | 51.89 |
| 14 | `2026-03-19_10L_MixedPrecision` | 10L Mixed Precision | Nan Liu | **1.21475** | 2.05105 | int8/int6 mixed | zlib | 15,928,974 | 13,101 | 45.78 |
| 15 | `2026-03-18_FP16Embed_WD3600` | FP16 Tied Embedding + LR/Warmdown Tuning | Renier Velazco | **1.21973** | 2.05945 | int8 (fp16 emb) | zlib | 15,896,222 | 13,692 | 43.82 |
| 16 | `2026-03-18_LowerLR` | Lower LR | Nan Liu | **1.22297** | 2.06493 | int8 | zlib | 15,854,246 | 14,421 | 41.60 |
| 17 | `2026-03-17_NaiveBaseline` | Naive Baseline | OpenAI | **1.22437** | 2.07270 | int8 | zlib | 15,863,489 | 13,780 | 43.54 |

**Total BPB improvement from baseline to SOTA: 0.0816 (6.7% relative)**

---

## Detailed Analysis Per Record

---

### 1. 10L Int5-MLP + BigramHash(10240) + SWA (CURRENT SOTA)
**Folder:** `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
**Author:** thwu1 | **Date:** 2026-03-20 | **val_bpb: 1.14276** (mean of 3 seeds, std 0.00016)

**Architecture:**
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2
- SmearGate + BigramHash(10240 buckets, dim=128)
- U-Net skip connections, tied embeddings
- Orthogonal init with muP-scaled output projections

**Training Config:**
- train_seq_len=2048, train_batch_tokens=786,432
- MATRIX_LR=0.02, WD=0.04, Muon momentum=0.99
- warmdown=3000, warmup=20, grad_clip=0.3
- SWA: start_frac=0.4, every=50 steps (24 checkpoints)
- 3% magnitude pruning
- ~6,711 steps in 600s at 89.42 ms/step

**Quantization:**
- **Mixed int5/int6**: int5 [-16,15] for MLP weights, int6 [-32,31] for attention weights
- FP16 for tied embeddings and last-layer key projections
- zstd-22 compression
- Int5 MLP saves ~1.86MB vs uniform int6, funding the 10th layer

**Convergence (seed 1337):**
| Step | val_bpb |
|------|---------|
| 500 | 1.3914 |
| 1000 | 1.3201 |
| 2000 | 1.2645 |
| 3000 | 1.2396 |
| 5000 | 1.2010 |
| 5500 | 1.1873 |
| 6000 | 1.1735 |
| 6500 | 1.1574 |
| 6711 (final) | 1.1533 (pre-quant) |
| **Final sliding** | **1.1430** |

**Multi-seed:** 42: 1.14271, 1337: 1.14298, 2024: 1.14260

**Novel Contributions:**
- First use of **int5 quantization** for MLP layers (5-bit, 32 levels) -- saves enough bytes to add a 10th layer
- Increased BigramHash from 4096 to 10240 buckets (-0.001 bpb improvement)
- SWA start_frac=0.4 (only average the most converged 40% of warmdown checkpoints)
- Built on PR #162 by @unnir

**Ablation:**
| Change | val_bpb | Delta |
|--------|---------|-------|
| 9L int6 base | 1.1485 | baseline |
| + int5 MLP + 10th layer | 1.1453 | -0.003 |
| + WD=0.04 + warmdown=3000 | 1.1452 | -0.0001 |
| + SWA_start_frac=0.4 | 1.1446 | -0.0006 |
| + bigram=10240 | **1.1426** | -0.002 |

---

### 2. Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Muon WD + SWA
**Folder:** `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`
**Author:** Raahil Shah | **Date:** 2026-03-20 | **val_bpb: 1.14582** (mean of 3 seeds, std 0.00082)

**Architecture:**
- 9 layers, 512 dim, 8 heads, 4 KV heads
- MLP 3x (hidden=1536), tied embeddings
- SmearGate (~512 params), BigramHash (4096 buckets, dim=128)
- Orthogonal weight init with muP output scaling

**Training Config:**
- train_seq_len=2048, train_batch_tokens=786,432
- MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03
- Muon WD=0.04, AdamW WD=0.01, Muon momentum=0.99 (warmup 0.92->0.99/1500 steps)
- warmdown=3000, grad_clip=0.3
- SWA every 50 steps over last 50% of training (~30 checkpoints)
- 7,379 steps in 600s at 81.3 ms/step

**Quantization:**
- Per-row int6 on MLP and attention weights, FP16 tied embeddings
- Last layer key projection in FP16
- zstd-22 compression
- Artifact: 15,862,650 bytes

**Pre-quant vs Post-quant:** 1.1616 vs 1.1458 => quant gap = **0.016 bpb** (int6 is lossy but SWA smoothes it)

**Convergence (seed 1337):**
| Step | val_bpb |
|------|---------|
| 500 | 1.4077 |
| 1000 | 1.3292 |
| 2000 | 1.2709 |
| 3000 | 1.2458 |
| 4000 | 1.2368 |
| 5000 | 1.2190 |
| 6000 | 1.1976 |
| 6500 | 1.1839 |

**Novel Contributions:**
- First combined deployment of SmearGate + BigramHash + SWA + int6 in a single submission
- Demonstrated SWA optimal sweep (every 25-200 steps, optimal at 50)
- Muon WD sweep (0.01-0.05, optimal at 0.04)

---

### 3. 11L MLP3x + Int6 QAT + zstd-22 + Sliding Window
**Folder:** `2026-03-19_MLP3x_QAT_Int6_SlidingWindow`
**Author:** aruniyer | **Date:** 2026-03-20 | **val_bpb: 1.15015** (mean of 3 seeds, std 0.00043)

**Architecture:**
- **11 layers** (most layers of any submission), 512 dim, 8 heads, 4 KV heads
- MLP 3x (hidden=1536), tied embeddings
- U-Net skip connections (5 encoder + 6 decoder)
- 26.5M parameters (largest model)

**Training Config:**
- train_seq_len=1024, train_batch_tokens=524,288
- MATRIX_LR=0.025, SCALAR_LR=0.025, TIED_EMBED_LR=0.035
- Muon WD=0.04, Muon momentum=0.99 (warmup 0.92->0.99/1500 steps)
- warmdown=3000
- QAT enabled (int6 STE fake-quantize during training)
- ~10,070 steps in 600s at 59.6 ms/step

**Quantization:**
- Int6 QAT on all block weights (layers 0-10), FP16 tied embeddings
- zstd-22 compression
- Artifact: ~15,427,455 bytes (smallest among top contenders)

**Key insight:** QAT allows training at seq_len=1024 (faster steps, more total steps) while still getting high quality from int6 quantization. No need for seq_len=2048.

**Convergence (seed 1337):**
| Step | val_bpb |
|------|---------|
| 6000 | 1.2464 |
| 12000 | 1.2053 |
| 12329 (final pre-quant) | 1.2007 |
| **Roundtrip standard** | **1.1998** |
| **Sliding window** | **1.1666** |

**Novel Contributions:**
- Most layers in any submission (11)
- QAT-enabled int6 eliminates quant gap (roundtrip = pre-quant effectively)
- zstd-22 critical for fitting 11-layer model under 16MB

---

### 4. SmearGate + OrthoInit + Muon WD
**Folder:** `2026-03-19_smeargate_orthoinit_muonwd`
**Author:** (unnir) | **Date:** 2026-03-19 | **val_bpb: 1.1556** (single seed)

**Architecture:**
- 9 layers, 512 dim, 8 heads, 4 KV heads
- MLP 3x (hidden=1536), tied embeddings
- SmearGate: learned per-dim gate blending current + previous token embedding
- BigramHash: 4096 buckets (dim=128, projected to 512)
- U-Net skip connections, orthogonal weight init
- 22,368,840 parameters

**Training Config:**
- train_seq_len=1024, train_batch_tokens=524,288
- MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03
- Muon WD=0.01, Muon momentum=0.99 (warmup 0.92->0.99/1500 steps)
- warmdown=3000
- Int6 STE QAT
- 12,047 steps in 600s at 49.80 ms/step

**Quantization:**
- Int6 QAT all block weights, FP16 tied embeddings
- zstd-22 compression
- Artifact: 15,878,809 bytes
- **Quantization gap: ~0.0001 BPB** (essentially zero thanks to QAT)

**Key metrics:**
- Post-quant standard val_bpb: 1.1891
- Post-quant sliding val_bpb: 1.1556
- Sliding window improvement: -0.0335 bpb

**Novel Contributions:**
- **Introduced SmearGate** -- lightweight bigram context injection at the embedding layer
- **Introduced BigramHash embedding** -- hash-based bigram feature table
- **Orthogonal weight initialization** -- all singular values = 1 at init, aligning with Muon's orthogonalization
- Combined OrthoInit + Muon WD for the first time

---

### 5. Int6 MLP3x Sliding Window (WarmdownQuantization)
**Folder:** `2026-03-19_WarmdownQuantization`
**Author:** samuellarson | **Date:** 2026-03-20 | **val_bpb: 1.15744**

**Architecture:**
- Appears to be MLP 3x with wider architecture (exact layer count unclear from README title says "Int6 MLP3x")
- train@2048 + sliding window eval + FP16 tied embeddings + Late-K passthrough
- ~83.35 ms/step suggests larger model

**Training Config:**
- 7,199 steps in 600s at 83.35 ms/step
- Sliding window eval stride=256, seq_len=2048

**Quantization:**
- Int6 post-training quantization (not QAT)
- zlib compression (not zstd)
- Artifact: 15,977,717 bytes (close to 16MB cap)

**Key metrics:**
- Pre-quant val_bpb: 1.1727
- Post-quant standard: 1.1789
- Post-quant sliding (stride=256): **1.15744**
- Quant gap: ~0.006 bpb

**Novel Contributions:**
- Explored aggressive warmdown (WARMDOWN_ITERS=20000) where entire training is in decay phase
- Found that warmdown dramatically reduces quantization penalty
- NTK-RoPE extrapolation for eval at seq_len=2048 when training used different lengths
- "Late-K passthrough" for preserving last-layer key projections in FP16

---

### 6. 10L Int6 QAT + Zstd MLP2.6x (Seq2048_FP16Emb_TunedLR)
**Folder:** `2026-03-19_Seq2048_FP16Emb_TunedLR`
**Author:** yahya010 | **Date:** 2026-03-19 | **val_bpb: 1.15862** (mean of 3 seeds, std 0.00120)

**Architecture:**
- 10 layers, 512 dim, 8 heads, 4 KV heads
- MLP hidden=1344 (2.625x expansion)
- Tied embeddings, FP16 embedding passthrough

**Training Config:**
- train_seq_len=2048
- MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.04
- Muon momentum=0.99, grad_clip=0.3
- Int6 STE QAT during training
- 8,319 steps in 600s at 72.13 ms/step

**Quantization:**
- Full int6 [-31,31] on all block weights, FP16 embedding
- zstd-22 compression
- Artifact: 15,558,319 bytes
- **Quant gap: 0.0000** -- QAT completely eliminated it

**Multi-seed:** 1337: 1.1610, 42: 1.1598, 3: 1.1586 (sliding)

**Novel Contributions:**
- Demonstrated that STE QAT can **completely eliminate** the quantization gap (0.0000 bpb)
- 2.625x MLP expansion (novel middle ground between 2x and 3x)
- QAT overhead only ~28% (72ms vs ~56ms without)

---

### 7. int6 STE QAT + MLP + Bigram + U-Net (INVALID -- OVER 16MB)
**Folder:** `2026-03-19_int6_STE QAT_ MLP_bigram _U_Net`
**Author:** unknown (no submission.json, no README) | **val_bpb: 1.15978** (sliding)

**Architecture:**
- 9 layers, 512 dim, 8 heads, 4 KV heads, MLP 3x
- 22,368,328 parameters

**Artifact: 16,186,500 bytes -- EXCEEDS 16MB cap. Invalid submission.**

**Metrics:**
- Pre-quant val_bpb: 1.1939
- Post-quant standard: 1.1931 (quant gap ~0.0008)
- Sliding window: 1.15978
- 12,123 steps at 49.49 ms/step

---

### 8. Mixed Quant int6/int8 + Sliding Window
**Folder:** `2026-03-19_MixedQuant_Int6Int8_SlidingWindow`
**Author:** aquariouseworkman | **Date:** 2026-03-19 | **val_bpb: 1.16301**

**Architecture:**
- 9 layers, 512 dim, 8 heads, 4 KV heads
- MLP 3x (hidden=1536)
- ~21.8M parameters

**Training Config:**
- train_seq_len=1024, train_batch_tokens=524,288
- MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03
- Muon momentum=0.99
- STE int6 fake-quantize during training (block weights only)
- 12,395 steps at 48.41 ms/step

**Quantization:**
- **Mixed precision**: int6 per-row (31 levels) on STE-protected block weights, int8 per-row (127 levels) on embedding
- zlib-9 compression
- Artifact: 15,353,490 bytes (efficient!)
- Quant penalty: +0.0015 BPB (32x reduction vs uniform int8 baseline)

**Key metrics:**
- Pre-quant val_bpb: 1.1950
- Post-quant standard: 1.1965
- Sliding window: **1.16301**

**Novel Contributions:**
- **Introduced mixed int6/int8 quantization** -- int6 for STE-trained weights, int8 for embeddings
- Showed that int8 on untrained-for-quantization embeddings is better than int6
- Detailed improvement breakdown showing each component's independent contribution

---

### 9. Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init
**Folder:** `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`
**Author:** notapplica | **Date:** 2026-03-19 | **val_bpb: 1.17475** (mean of 3 seeds)

**Architecture:**
- 10 layers, 512 dim (inferred from int8+zlib artifact and context)
- FP16 tied embedding

**Training Config:**
- ~10,424-10,710 steps at ~57 ms/step
- Muon WD=0.02
- Compiled forward_logits for efficient sliding window eval

**Quantization:**
- int8 + zlib (standard)
- FP16 embedding passthrough
- Artifact: ~15,374,243 bytes

**Multi-seed:** 1337: 1.1756, 42: 1.1742, 7: 1.1744

**Novel Contributions:**
- **Overtone spectral embedding init**: SVD power-law spectrum shaping (S_k ~ k^{-0.5})
- **Phase-transition residual mixing**: Sigmoid-scheduled resid_mix initialization
- Compiled sliding window eval for efficiency

---

### 10. LoRA TTT (Test-Time Training)
**Folder:** `2026-03-17_LoRA_TTT`
**Author:** sam (samacqua) | **Date:** 2026-03-19 | **val_bpb: 1.1929**

**Architecture:** Same as baseline (9x512, MLP 2x, tied embeddings)

**Training:** Identical to naive baseline

**Evaluation Innovation:**
- Per-document LoRA test-time training at eval
- Rank-8 LoRA on lm_head, Q, V projections
- Adam lr=0.01, betas=(0.9, 0.95)
- Overlapping 256-token chunks in 1024-token context windows
- Documents batched (batch_size=64), sorted by length
- Uses ~1/10th of eval budget

**Key ablation:**
| Condition | val_bpb | Delta |
|-----------|---------|-------|
| Baseline | 1.2278 | -- |
| + Doc-isolated | 1.2168 | -0.0110 |
| + Stride (chunk=256) | 1.1941 | -0.0337 |
| + LoRA TTT | 1.1910 | -0.0368 |

**Novel Contributions:**
- First TTT submission -- most improvement comes from doc-isolation and striding, not TTT itself
- Showed LoRA makes TTT fast enough for batched eval

---

### 11. Sliding Window Eval (stride=64)
**Folder:** `2026-03-19_SlidingWindowEval`
**Author:** Matthew Li (mattqlf) | **Date:** 2026-03-19 | **val_bpb: 1.19250**

**Architecture:** Identical to baseline (9x512, MLP 2x)

**Training:** Identical to naive baseline. 13,450 steps at 44.61 ms/step.

**Evaluation Innovation:**
- Sliding window with stride=64 (every token scored with 960+ context)
- Eval time: 70s (vs ~16s baseline)
- **Pure eval improvement: -0.0319 bpb with zero training changes**

**Convergence (standard eval, pre-quant):**
| Step | val_bpb |
|------|---------|
| 1000 | 1.3823 |
| 3000 | 1.2994 |
| 5000 | 1.2749 |
| 7000 | 1.2626 |
| 9000 | 1.2540 |
| 11000 | 1.2472 |
| 13000 | 1.2286 |
| 13450 | 1.2196 (pre-quant) |
| **Post-quant sliding** | **1.1925** |

**Novel Contributions:**
- **Introduced sliding window evaluation** to the competition
- Proved that eval strategy alone is worth -0.032 bpb
- Clean isolation of technique shows this is "free" improvement

---

### 12. Training Opt Seq4096 v1
**Folder:** `2026-03-19_TrainingOptSeq4096`
**Author:** Spokane Way | **Date:** 2026-03-19 | **val_bpb: 1.20143**

**Architecture:** Baseline (9x512, MLP 2x)

**Training Config:**
- train_seq_len=4096 (4x baseline)
- train_batch_tokens=393,216 (3/4 batch for more optimizer steps)
- MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03
- Muon momentum=0.99 (warmup 0.92->0.99/1500 steps)
- warmdown=3000
- 8,394 steps at 71.47 ms/step

**Quantization:** Standard int8 + zlib. Quant gap: 0.0034 bpb (reduced by lower LR)

**Convergence:**
| Step | val_bpb |
|------|---------|
| 2000 | 1.3095 |
| 4000 | 1.2557 |
| 6000 | 1.2309 |
| 8000 | 1.2015 |
| 8394 | 1.1980 (pre-quant) |
| **Post-quant** | **1.2014** |

**Novel Contributions:**
- Demonstrated value of long-context training (4096 tokens)
- Systematic Muon optimizer tuning (momentum 0.99, lower LR, 3/4 batch)
- Extended momentum warmup prevents instability at higher momentum

---

### 13. Long Context Seq2048 v2
**Folder:** `2026-03-18_LongContextSeq2048`
**Author:** Spokane Way | **Date:** 2026-03-19 | **val_bpb: 1.20576**

**Architecture:** Baseline (9x512, MLP 2x)

**Training Config:**
- train_seq_len=2048 (2x baseline)
- train_batch_tokens=524,288
- LRs: TIED_EMBED_LR=0.04, MATRIX_LR=0.032, SCALAR_LR=0.032
- 11,564 steps at 51.89 ms/step

**Quantization:** Standard int8 + zlib. Quant gap: 0.0053 bpb

**Convergence:**
| Step | val_bpb |
|------|---------|
| 1000 | 1.3568 |
| 3000 | 1.2748 |
| 5000 | 1.2507 |
| 7000 | 1.2382 |
| 9000 | 1.2298 |
| 11000 | 1.2110 |
| 11564 | 1.2005 (pre-quant) |
| **Post-quant** | **1.2058** |

**Novel Contributions:**
- First seq_len=2048 submission showing longer context helps training quality
- 3 seed verification with statistical significance testing

---

### 14. 10L Mixed Precision
**Folder:** `2026-03-19_10L_MixedPrecision`
**Author:** Nan Liu (nanlliu) | **Date:** 2026-03-19 | **val_bpb: 1.21475**

**Architecture:** 10 layers, 512 dim, 8 heads, 4 KV heads, MLP 2x

**Training Config:**
- MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03
- 13,101 steps at 45.78 ms/step

**Quantization:**
- Mixed: int8 for first/last 3 layers (0-2, 7-9), int6 for middle layers (3-6)
- zlib compression
- Quant gap: 0.0018 bpb

**Convergence:**
| Step | val_bpb |
|------|---------|
| 200 | 1.6170 |
| 600 | 1.4315 |
| 1000 | 1.3677 |
| 12800 | 1.2129 |
| 13101 | 1.2080 (pre-quant) |
| **Post-quant** | **1.2147** |

**Novel Contributions:**
- **First use of mixed int8/int6 compression** (by layer sensitivity)
- Proved middle layers can tolerate int6 with minimal quality loss
- Added 10th transformer layer funded by int6 compression savings

---

### 15. FP16 Tied Embedding + LR/Warmdown Tuning
**Folder:** `2026-03-18_FP16Embed_WD3600`
**Author:** Renier Velazco (chonchiog) | **Date:** 2026-03-18 | **val_bpb: 1.21973**

**Architecture:** 9 layers, 512 dim, MLP hidden=992 (reduced from 1024)

**Training Config:**
- MATRIX_LR=0.06, WARMDOWN_ITERS=3600
- 13,692 steps at 43.82 ms/step

**Quantization:**
- int8 + zlib, **FP16 embedding passthrough**
- Quant gap: ~0.0005 bpb (down from ~0.007 baseline)

**Novel Contributions:**
- **Introduced FP16 tied embedding passthrough** -- the key insight that the shared input/output embedding is the most quantization-sensitive tensor
- Reduced quant gap from 0.007 to 0.0005 bpb (14x reduction)
- Explored SwiGLU (better per-step but 45% slower, net negative)

---

### 16. Lower LR
**Folder:** `2026-03-18_LowerLR`
**Author:** Nan Liu (nanlliu) | **Date:** 2026-03-18 | **val_bpb: 1.22297**

**Architecture:** Baseline (9x512, MLP 2x)

**Training Config:**
- MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03
- 14,421 steps at 41.60 ms/step (on H200)

**Quantization:** Standard int8 + zlib. Quant gap: 0.0047 bpb

**Novel Contributions:**
- Systematic LR sweep (0.015 to 0.06) showing default 0.04 is too high
- Optimal at ~0.02 (half default)
- Note: run on H200, which is ~5% faster than H100

---

### 17. Naive Baseline
**Folder:** `2026-03-17_NaiveBaseline`
**Author:** OpenAI | **Date:** 2026-03-18 | **val_bpb: 1.22437**

**Architecture:** 9 layers, 512 dim, 8 heads, 4 KV heads, MLP 2x, tied embeddings, 1024 BPE vocab

**Training Config:**
- train_seq_len=1024, train_batch_tokens=524,288
- Default LRs (0.04/0.04/0.05)
- 13,780 steps at 43.54 ms/step

**Quantization:** Standard int8 + zlib. Quant gap: 0.0072 bpb

**Convergence:**
| Step | val_bpb |
|------|---------|
| 200 | 1.6774 |
| 400 | 1.5174 |
| 600 | 1.4447 |
| 800 | 1.4099 |
| 1000 | 1.3805 |
| 2000 | 1.3213 |
| 5000 | 1.2839 |
| 10000 | 1.2471 |
| 13780 | 1.2172 (pre-quant) |
| **Post-quant** | **1.2244** |

---

## Technique Evolution Timeline

| Date | Key Innovation | BPB Impact |
|------|---------------|------------|
| Mar 17 | Naive Baseline | 1.2244 (reference) |
| Mar 18 | FP16 embedding passthrough | -0.005 (quant gap reduction) |
| Mar 18 | Lower LR (0.02 vs 0.04) | -0.002 |
| Mar 18 | Longer context (seq=2048) | -0.019 |
| Mar 19 | Sliding window eval (stride=64) | -0.032 (free, eval-only) |
| Mar 19 | Muon momentum 0.99 + warmup | -0.005 |
| Mar 19 | MLP 3x + int6 quantization | -0.028 (more params in same bytes) |
| Mar 19 | STE QAT int6 | eliminates quant gap (0.000) |
| Mar 19 | zstd-22 over zlib | -1.5MB savings (enables more params) |
| Mar 19 | SmearGate + BigramHash | -0.005 (bigram context injection) |
| Mar 19 | Orthogonal init | -0.002 (faster convergence) |
| Mar 19 | U-Net skip connections | ~-0.001 |
| Mar 19 | Muon weight decay | -0.003 (regularization + quant friendliness) |
| Mar 20 | SWA (Stochastic Weight Averaging) | -0.003 |
| Mar 20 | Int5 MLP quantization | enables 10th layer (-0.003) |
| Mar 20 | BigramHash 10240 buckets | -0.001 |

## Key Patterns and Insights

1. **Sliding window eval is the single biggest "free" win**: ~0.032 bpb improvement with zero artifact or training cost. Every submission from #1-#8 uses it.

2. **Quantization innovations dominate the leaderboard**: The progression from int8->int6->int5 with QAT, mixed precision, and FP16 embedding passthrough accounts for the largest improvements. Better quantization means more parameters in 16MB.

3. **MLP 3x is critical**: Widening the MLP from 2x to 3x was funded by int6 quantization and is one of the largest single improvements (~0.03 bpb).

4. **Training time per step tradeoff**: The best models run at 80-90 ms/step (vs 43 ms baseline), sacrificing step count for per-step quality (wider MLP, longer seq, QAT overhead). The sweet spot is ~7000-8000 high-quality steps rather than ~14000 fast steps.

5. **Compression evolution**: zlib -> zstd-22 saves ~1.5MB, which translates directly to more model parameters. All top-5 entries use zstd-22.

6. **Embedding techniques matter at the margin**: SmearGate and BigramHash add ~0.005 bpb combined with minimal parameter overhead. They appear in all top-4 entries.

7. **SWA smooths quantization**: Averaging checkpoints from the converged portion of training produces weight distributions that quantize better, a surprising synergy.
