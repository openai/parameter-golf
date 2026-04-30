# Complete Parameter Golf Techniques Mapping
## All Local Submissions + Top 10 GitHub PRs

**Summary**: Comprehensive analysis of all 34 submissions (24 local + 10 GitHub PRs) with technique mapping.

**Generated**: 2026-04-14

**Coverage**:
- ✅ 21 record submissions (local)
- ✅ 3 non-record submissions (local)
- ✅ 10 top GitHub PRs (non-record/experimental)

---

## Technique Adoption Summary

### By Local Submissions (24 total)

| Rank | Technique | Count | % | Examples |
|------|-----------|-------|---|----------|
| 1 | **Muon Optimizer** | 23 | 96% | All submissions use Muon |
| 2 | **Int6 Quantization** | 23 | 96% | Core quantization approach |
| 3 | **EMA (Exponential Moving Avg)** | 23 | 96% | Weight regularization |
| 4 | **GQA (Grouped Query Attention)** | 23 | 96% | Attention efficiency |
| 5 | **Cosine Warmdown** | 23 | 96% | Learning rate schedule |
| 6 | **Sliding Window Eval** | 15 | 62% | Eval-time trick (+0.019 BPB) |
| 7 | **SWA (Stochastic Weight Averaging)** | 9 | 38% | Ensemble averaging |
| 8 | **BigramHash** | 9 | 38% | Low-cost embedding table |
| 9 | **XSA (Cross-Sequence Attention)** | 7 | 29% | Cross-position mixing |
| 10 | **GPTQ** | 2 | 8% | Post-train quantization (SOTA) |

### By Top 10 GitHub PRs

| Technique | PRs | Examples |
|-----------|-----|----------|
| **Flash Attention** | 6 | Efficient attention computation |
| **Diffusion** | 2 | Alternative generative approach |
| **JEPA** | 2 | Joint Embedding Predictive Architecture |
| **U-Net** | 1 | Alternative architecture (PR #1577) |
| **Depth Recurrence** | 3 | Shared block approach (PR #386) |
| **LoRA** | 4 | Low-rank adaptation |

---

## Local Submissions: Technique → Submissions Mapping

### Foundational (96% adoption)

#### Muon Optimizer
**Found in**: All 23 record submissions + 1 non-record

The Muon optimizer with parallel Newton-Schulz orthogonalization is the baseline for all Parameter Golf attempts. Key hyperparameters:
- `muon_momentum`: 0.95–0.99
- `muon_backend_steps`: 5–10
- `muon_wd`: 0.04–0.09

**Ranked submissions using Muon**:
- 2026-03-25: ValCalib GPTQ + XSA + BigramHash (1.1147 BPB)
- 2026-03-22: 11L EMA GPTQ-lite + warmdown (1.1233 BPB)
- 2026-03-21: 11L XSA4 EMA PartialRoPE LateQAT (1.1248 BPB)
- 2026-03-20: 11L XSA4 EMA Int6 MLP3x WD04 (1.1271 BPB)

---

#### Int6 Quantization (QAT)
**Found in**: 23 submissions (96%)

Straight-through estimator (STE) quantization-aware training enables ~70M effective params in 16MB.

**Adoption pattern**:
- Early (03-17 to 03-19): Int8 baseline, mixed Int4/Int6/Int8
- Mid (03-20 to 03-22): Int6 dominates (better than Int8)
- Late (03-25+): Full Int6 + GPTQ for SOTA

**Ranked submissions**:
- 2026-03-20: Int6 + SmearGate + BigramHash + Muon WD
- 2026-03-19: MLP3x QAT Int6 SlidingWindow
- 2026-03-19: MixedQuant Int6Int8 SlidingWindow (still experimenting)

---

#### EMA (Exponential Moving Average)
**Found in**: 23 submissions (96%)

`ema_decay=0.997` applied to all weights post-backward.

**Improvement**: ~0.003 BPB from checkpoint averaging.

**Ranked submissions using EMA**:
- 2026-03-22: 11L EMA GPTQ-lite (1.1233 BPB)
- 2026-03-21: 11L XSA4 EMA PartialRoPE (1.1248 BPB)
- 2026-03-20: 11L XSA4 EMA Int6 MLP3x (1.1271 BPB)

---

#### GQA (Grouped Query Attention)
**Found in**: 23 submissions (96%)

Using `num_kv_heads=4` (vs 8 heads) reduces memory and params.

**Standard configuration**:
- `num_heads=8`
- `num_kv_heads=4`
- `num_layers=9–11`

All rank submissions use this.

---

#### Cosine Warmdown
**Found in**: 23 submissions (96%)

Cosine annealing learning rate from warmup peak → 0 over final epochs.

**Common schedules**:
- `warmdown_iters=3500–4000` for 20k total iterations
- Pair with `warmup_steps=20`

---

### High Impact (30–70% adoption)

#### Sliding Window Eval
**Found in**: 15 submissions (62%)

Shift evaluation window at stride=64 for free ~0.019 BPB gain.

**How it works**:
- Standard: eval on fixed window [0:context_len]
- Sliding: eval on [0:64], [64:128], [128:192], ... average loss

**Ranked submissions with Sliding Window**:
- 2026-03-25: ValCalib GPTQ + XSA + BigramHash (1.1147 BPB)
- 2026-03-19: SlidingWindowEval baseline
- 2026-03-19: SlidingWindow + FP16Embed
- 2026-03-19: MixedQuant Int6Int8 SlidingWindow

---

#### SWA (Stochastic Weight Averaging)
**Found in**: 9 submissions (38%)

Every N steps, add current weights to running average. Use averaged weights at eval.

**Common hyperparameters**:
- `swa_every=50–120` steps
- `swa_enabled=True/False`

**Improvement**: ~0.001–0.002 BPB from ensemble-like effects.

**Ranked submissions**:
- 2026-03-20: 11L EfficientPartialXSA + FA3 SWA120 (1.1271 BPB)
- 2026-03-20: Int6 MLP3x SmearGate BigramHash Muon WD SWA
- 2026-03-20: 10L Int5MLP Muon WD04 SWA50

---

#### BigramHash
**Found in**: 9 submissions (38%)

Hash consecutive token pairs into embedding table (3072×112 in SOTA) at low parameter cost.

**Configuration in SOTA**:
- `bigram_vocab_size=3072`
- `bigram_dim=112`
- ~340K params for massive co-occurrence coverage

**Improvement**: ~0.004 BPB from local syntax patterns.

**Ranked submissions**:
- 2026-03-25: ValCalib GPTQ + XSA + **BigramHash 3072** (1.1147 BPB) ⭐ **SOTA**
- 2026-03-20: Int6 MLP3x SmearGate + BigramHash + Muon WD
- 2026-03-20: 11L EfficientPartialXSA FA3 SWA (BigramHash)
- 2026-03-19: Int6 STE QAT MLP Bigram U-Net

---

#### XSA (Cross-Sequence Attention)
**Found in**: 7 submissions (29%)

Cross-position information mixing applied to all 11 layers (novel in SOTA).

**Configuration**:
- `xsa_last_n=11` (applies to all layers, not just last)
- Zero additional parameters
- Free mixing of cross-position information

**Improvement**: ~0.003 BPB from better position mixing.

**Ranked submissions**:
- 2026-03-25: ValCalib GPTQ + **XSA**-all + BigramHash (1.1147 BPB) ⭐ **SOTA**
- 2026-03-21: 11L **XSA**4 EMA PartialRoPE LateQAT (1.1248 BPB)
- 2026-03-20: 11L **XSA**4 EMA Int6 MLP3x WD04 (1.1271 BPB)
- 2026-03-20: 11L EfficientPartialXSA FA3 SWA120 (1.1271 BPB)

---

### Advanced (≤12% adoption)

#### GPTQ (Post-Training Quantization)
**Found in**: 2 submissions (8%)

Post-training Hessian-approximated quantization. AR self-generated calibration (no external data).

**Current SOTA approach**:
- Full Hessian computation per block
- Calibration: 256 batches of FineWeb training data
- Applied before LZMA compression
- ~0.005 BPB improvement over raw Int6

**SOTA submission**:
- 2026-03-25: **ValCalib GPTQ** + XSA + BigramHash (1.1147 BPB) ⭐

**Other use**:
- 2026-03-22: 11L EMA **GPTQ**-lite + warmdown (1.1233 BPB)

---

#### LZMA Compression
**Found in**: 2 submissions (8%)

LZMA preset=9 replaces zlib for model bytes compression.

**Trade-off**:
- Better compression ratio (vs zlib)
- ~30s extra decompression time during eval
- Worth it for SOTA when every byte counts

**Submissions**:
- 2026-03-25: ValCalib GPTQ + XSA + BigramHash (uses LZMA for final compression)
- 2026-03-22: GPTQ-lite (LZMA)

---

#### TTT (Test-Time Training)
**Found in**: 2 submissions (8%)

Train adaptation parameters at test time using current sequence.

**Variants found**:
- LoRA TTT (rank 4–8 on query/value projections)
- Legal TTT (pre-eval training, ~1% of budget)

**Submissions**:
- 2026-03-23: LeakyReLU **LegalTTT** ParallelMuon
- 2026-03-17: LoRA **TTT** (baseline experiment)

**Status**: Dropped from SOTA (violates 10min compute constraint for full effect).

---

#### SmearGate
**Found in**: 8 submissions (33%)

Gating mechanism for attention values.

**Submissions**:
- 2026-03-20: Int6 MLP3x **SmearGate** BigramHash Muon WD SWA
- 2026-03-19: smeargate + orthoinit + muonwd

---

#### PartialRoPE
**Found in**: 4 submissions (17%)

Rotary positional embeddings on subset of dims (vs all dims).

**Configuration**:
- `rope_dims=4–16` (not full model_dim)
- Reduces parameter cost of position encoding

**Submissions**:
- 2026-03-21: 11L XSA4 EMA **PartialRoPE** LateQAT (1.1248 BPB)
- 2026-03-19: SlidingWindow + FP16Emb + 10L Muon WD OvertoneInit

---

#### OrthoInit (Orthogonal Initialization)
**Found in**: 2 submissions (8%)

Initialize weight matrices to be orthogonal (better conditioning).

**Submissions**:
- 2026-03-19: smeargate + **orthoinit** + muonwd
- 2026-03-19: SlidingWindow FP16Emb Muon WD **Overtone**Init

---

#### LeakyReLU
**Found in**: 2 submissions (8%)

Alternative activation function (vs ReLU or GELU).

**Submissions**:
- 2026-03-23: **LeakyReLU** LegalTTT ParallelMuon

---

---

## Top 10 GitHub PRs: New Techniques

### Alternative Architectures (Non-Record Explorations)

#### JEPA (Joint Embedding Predictive Architecture)
**PRs**: #1480

**Concept**: Pre-training approach that predicts latent representations instead of tokens.

**Score**: 1.2699 BPB (non-record, over 10min)

**Techniques combined**:
- EMA student/teacher networks
- Masked latent prediction
- Custom tokenizer (sp1024)

---

#### GDN (Gated DeltaNet)
**PRs**: #875, #1576

**Concept**: Gated Delta Networks — alternative RNN-like architecture.

**Scores**:
- PR #875: 1.0226 BPB (pure neural, test-time free)
- PR #1576: GDN-Hybrid + Sliding Window

**Why explored**: Better capacity per parameter than attention (in theory), but doesn't compress well.

---

#### U-Net Architecture
**PRs**: #1577

**Concept**: Encoder-decoder U-Net for sequence modeling.

**Score**: 1.40 BPB

**Structure**:
- Byte-level encoder (256 vocab, no BPE)
- Skip connections between encoder/decoder
- Patched latent representations

---

#### Masked Text Diffusion
**PRs**: #1596

**Concept**: Diffusive generative approach (vs autoregressive).

**Techniques**:
- Masked diffusion loss
- SP1024 tokenizer
- Multi-step generation

**Status**: Explored but not competitive vs. autoregressive + quantization.

---

#### LoRA (Low-Rank Adaptation)
**PRs**: #468, #517, #647, others

**Concept**: Fine-tune via low-rank matrices instead of full weights.

**Common configurations**:
- LoRA rank: 4–16
- Applied to Q, V projections
- Can combine with QAT

---

#### Depth Recurrence
**PRs**: #386

**Concept**: Shared transformer block applied multiple times (≈12 passes).

**Benefit**: Massive parameter reduction.

**Trade-off**: Slower forward pass, less expressive.

---

---

## Architecture Landscape

### Top Techniques by Effectiveness Tier

**Tier 1 (Must-Have)**: ~0.05 BPB impact
- Muon optimizer (foundational)
- Int6 QAT
- GQA
- Cosine warmdown

**Tier 2 (High Impact)**: ~0.01–0.02 BPB each
- EMA
- Sliding window eval
- SWA
- BigramHash
- XSA

**Tier 3 (Niche/Risky)**: ~0.001–0.005 BPB
- PartialRoPE
- FP16 embedding
- OrthoInit
- SmearGate
- LZMA compression
- NTK scaling

**Tier 4 (Experimental)**: Unproven in constraint
- JEPA
- Diffusion
- GDN
- U-Net
- TTT (violates constraints)

---

## Local Submission Rankings (by BPB)

| Rank | Submission | BPB | Key Techniques | Track |
|------|-----------|-----|-----------------|-------|
| 1 | 2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072 | **1.1147** | GPTQ, XSA, BigramHash, EMA, SWA, sliding window | record |
| 2 | 2026-04-01_Vocab4096_MLPMult4_WD085 | 1.1148 | Larger vocab, MLP 4×, weight decay | record |
| 3 | 2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271 | 1.1271 | XSA, EMA, Int6, MLP 3× | record |
| 4 | 2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233 | 1.1233 | GPTQ-lite, EMA, late QAT | record |
| 5 | 2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248 | 1.1248 | XSA, PartialRoPE, late QAT | record |

---

## Technique Evolution Timeline

```
2026-03-17 Baseline: Muon + Int6 QAT + Cosine warmdown
             ↓
2026-03-18 +Sliding Window Eval (+0.019 BPB)
             ↓
2026-03-19 +EMA, SWA, FP16 Embed, PartialRoPE (+0.008 BPB)
             ↓
2026-03-20 +BigramHash 3072×112 (+0.004 BPB)
           +XSA on all layers (+0.003 BPB)
             ↓
2026-03-21 +Late QAT (fine-grain Int6 after training) (+0.001 BPB)
             ↓
2026-03-22 +Full GPTQ (Hessian-aware quantization) (+0.005 BPB)
             ↓
2026-03-25 Synthesis: GPTQ + XSA + BigramHash + EMA + SWA + LZMA
           SOTA: 1.1147 BPB ⭐
             ↓
2026-04-01 Alternative: Larger vocab, MLP 4×, weight decay tuning
           SOTA: 1.1148 BPB (marginal alternative)
```

---

## Files Reference

- `docs/leaderboard-guide.md` — Full GitHub leaderboard map (60+ URLs)
- `docs/local-techniques-analysis.md` — Detailed local submission analysis
- `docs/top-prs-analysis.md` — GitHub PR techniques
- `docs/techniques-analysis.md` — BPB impact analysis of 25 techniques
- `docs/winning-techniques.md` — Tier strategy and attack vectors

---

**Last updated**: 2026-04-14 | **Status**: Complete (34/34 submissions analyzed)
