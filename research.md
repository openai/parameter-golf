# 🏆 Parameter Golf — Deep Research & Battle Plan

> **Challenge**: OpenAI Model Craft Challenge: Parameter Golf
> **Dates**: March 18 – April 30, 2026
> **Current SOTA**: **1.1428 val_bpb** (thwu1, Mar 20)
> **Our Target**: Beat 1.1428 by ≥ 0.005 nats → need **≤ 1.1378 val_bpb** (approx)

---

## 1. What Is This Challenge?

### The Core Objective
Train the **best language model** that:
1. **Fits in a 16MB artifact** (code bytes + compressed model bytes ≤ 16,000,000 bytes)
2. **Trains in ≤ 10 minutes** on 8×H100 SXM GPUs
3. **Minimizes val_bpb** (bits per byte) on the FineWeb validation set

### Scoring Metric: val_bpb (Bits Per Byte)
- **Tokenizer-agnostic** compression metric
- Measures how well the model compresses unseen text from FineWeb validation (first 50k documents)
- Lower is better
- Formula: `val_bpb = (val_loss / ln(2)) * (tokens_per_byte = token_count / byte_count)`

### Constraints
| Constraint | Value |
|---|---|
| **Artifact size cap** | 16,000,000 bytes (decimal 16 MB) |
| **Training time** | ≤ 10 minutes on 8×H100 SXM |
| **Evaluation time** | ≤ 10 minutes on 8×H100 SXM (additional) |
| **No external data** | No internet/network during training or eval |
| **No val data leaking** | Cannot train on validation data (TTT only on already-scored tokens) |
| **Significance** | Must beat SOTA by ≥ 0.005 nats at p < 0.01 (typically 3 seeds) |

### What Counts as the Artifact
- The `train_gpt.py` script (code bytes)
- The compressed model checkpoint (quantized + compressed weights)
- Everything must be self-contained and reproducible

---

## 2. The Baseline Architecture

From the [Naive Baseline](file:///d:/Personal/Projects/Code/parameter-golf/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md) (**1.2244 val_bpb**):

| Component | Value |
|---|---|
| **Layers** | 9 transformer blocks |
| **Model dim** | 512 |
| **Attention heads** | 8 (4 KV heads via GQA) |
| **MLP expansion** | 2× (hidden=1024), ReLU² activation |
| **Vocab size** | 1024 (SentencePiece BPE) |
| **Sequence length** | 1024 (training) |
| **Embeddings** | Tied input/output |
| **Skip connections** | U-Net style (encoder/decoder halves) |
| **Positional encoding** | RoPE (base=10000) |
| **Optimizer** | Muon (matrix params) + AdamW (scalars/embeddings) |
| **Quantization** | Post-training int8 + zlib |
| **Training tokens** | ~7.2B tokens in 13,780 steps × 524K tokens/step |
| **Step time** | ~43.5 ms/step on 8×H100 |
| **Logit softcap** | 30.0 |
| **Parameter count** | ~17.1M |

### Key Architecture Details
- **CastedLinear**: Weights in fp32, casted to bf16 for matmul (optimizer quality)
- **RMSNorm**: Applied before attention and MLP (pre-norm)
- **Resid Mix**: Learned mix between residual stream `x` and initial embedding `x0`
- **U-Net Skips**: First half of layers stores activations, second half adds them back via learned skip weights
- **Flash Attention**: cuDNN disabled, Flash SDP enabled

---

## 3. Leaderboard Evolution & Analysis

### Score Progression (Baseline → Current SOTA)

```
1.2244  ← Naive Baseline (9L, int8, standard eval)
1.2230  ← Lower LR (MATRIX_LR 0.04→0.02) — LR was too high
1.2197  ← FP16 Embed (tied emb kept in fp16, not int8) — huge quant gap reduction
1.2154  ← Warmdown Quantization (WARMDOWN_ITERS=20000, full-decay schedule)
1.2147  ← 10L Mixed Precision (int6 middle layers, 10 layers)
1.2060  ← Long Context Seq2048 (training at 2048 seq len)
1.2014  ← 4K Seq + Muon tuning (seq4096, momentum 0.99, lower LR)
1.1925  ← Sliding Window Eval (stride=64, FREE 0.032 bpb improvement!)
1.1928  ← LoRA TTT (test-time training with LoRA adapters)
1.1748  ← Muon WD + 10L + FP16 Emb + Sliding + Overtone Init
1.1630  ← Mixed Quant Int6/Int8 + MLP 3× + Sliding
1.1586  ← Int6 QAT + zstd + MLP 2.6× + Muon 0.99
1.1556  ← SmearGate + OrthoInit + Muon WD + BigramHash
1.1502  ← 11L MLP3x + Int6 QAT + zstd-22 + WD=0.04
1.1458  ← #2: Int6 MLP3x + SmearGate + BigramHash + OrthoInit + SWA
1.1428  ← #1 CURRENT SOTA: 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD=0.04
```

### Total BPB Improvement: **−0.0816 bpb** from baseline to SOTA

---

## 4. Techniques Catalog (What Has Been Tried & Their Impact)

### 4.1 Compression / Quantization Techniques

| Technique | BPB Impact | Description |
|---|---|---|
| **FP16 Tied Embedding** | −0.006 to −0.007 | Keep embeddings in fp16 instead of int8. Huge impact because embedding pulls double duty (input + output) |
| **Int6 Quantization** | enables MLP 3× | 63 levels instead of 255. Much more compressible with zstd. ~1.5× better compression ratio |
| **Int5 Quantization (MLP only)** | −0.003 (via 10th layer) | MLP weights tolerate int5 [-16,15]. Saves ~1.86MB → funds extra layer |
| **Int6 STE QAT** | eliminates quant gap | Fake-quantize weights during forward pass with Straight-Through Estimator. Model learns int6-robust weights |
| **zstd-22 Compression** | −1.5MB vs zlib | Much better than zlib-9 for int6 data. Critical for fitting more layers/params |
| **3% Magnitude Pruning** | minor | Remove smallest 3% of quantized values |
| **Mixed Precision Export** | varies | Different quant levels for different layers (early/late more sensitive) |

### 4.2 Architecture Techniques

| Technique | BPB Impact | Description |
|---|---|---|
| **MLP 3× Expansion** | −0.029 | Largest single contributor! Hidden dim 1024→1536. Enabled by int6 byte savings |
| **10+ Layers** | −0.003 to −0.005 | More layers (10L or 11L) funded by aggressive quantization |
| **SmearGate** | −0.002 to −0.003 | Learned gate blending current + previous token embedding. ~512 params |
| **BigramHash** | −0.003 to −0.005 | Hash table for token-pair embeddings. 4096-10240 buckets, dim=128 |
| **U-Net Skip Connections** | baseline | Already in baseline, connects encoder/decoder halves |
| **SwiGLU** | ❌ negative (8×GPU) | Better per-step but 45% slower → fewer total steps on 8×H100 |
| **Layer Recurrence** | ❌ (−0.051!) | Halves steps, which destroys progress. Worst technique tried |

### 4.3 Optimization Techniques

| Technique | BPB Impact | Description |
|---|---|---|
| **Lower LR (0.04→0.02)** | −0.006 | Default was too high. Optimal at 0.02 |
| **Muon Momentum 0.99** | −0.003 to −0.005 | Up from 0.95. Warmup from 0.92 over 1500 steps |
| **Weight Decay 0.01-0.04** | −0.002 to −0.005 | Decoupled WD on Muon. Helps generalization AND quantization |
| **Gradient Clipping 0.3** | minor | Prevents gradient explosions |
| **Orthogonal Init** | −0.002 | Uniform singular values → faster early convergence. Matches Muon's orthogonalization |
| **WARMDOWN_ITERS tuning** | −0.005 to −0.009 | Setting 3000+ makes LR decay properly within wallclock limit |
| **SWA (Stochastic Weight Averaging)** | −0.001 to −0.003 | Average checkpoints from last 40-50% of training. Better quantization |

### 4.4 Evaluation Techniques

| Technique | BPB Impact | Description |
|---|---|---|
| **Sliding Window Eval (stride=64)** | −0.032 to −0.034 | FREE improvement! Every token gets 960+ context tokens. ~70-90s eval time |
| **LoRA TTT (Test-Time Training)** | −0.004 to −0.006 | Per-document LoRA adaptation during eval. Rank-8, Adam lr=0.01 |
| **NTK-RoPE Extrapolation** | −0.007 (moderate) | Eval at 1.375× train seq. But overtrained models are sensitive to distortion |
| **Doc-Isolated Eval** | −0.011 | Only condition on current document, not cross-doc context |

### 4.5 Training Configuration

| Technique | BPB Impact | Description |
|---|---|---|
| **Seq Len 2048/4096** | −0.018 to −0.023 | Longer context → better quality. ~72ms/step at 4096 vs 43ms at 1024 |
| **Batch Tokens 786K** | mild | Allows MLP 3× with longer seq on 8×H100 |
| **Smaller Batch + More Steps** | −0.016 (1×GPU) | Trade batch for more optimizer updates. Hardware dependent |

---

## 5. What the Current SOTA Does (Anatomy of #1)

[thwu1's winning entry](file:///d:/Personal/Projects/Code/parameter-golf/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md): **1.14276 val_bpb**

### Architecture
- **10 layers**, 512 dim, 8 heads, 4 KV heads (GQA)
- **MLP 3× expansion** (hidden=1536), **ReLU² activation**
- **SmearGate** + **BigramHash(10240, dim=128)** — 10240 buckets!
- **Orthogonal init** with muP-scaled output projections
- **U-Net skip connections**, tied embeddings

### Quantization
- **Int5 [-16,15] for MLP weights** (1.88× zstd ratio — most compressible)
- **Int6 [-32,31] for attention weights** (precision-sensitive, 1.51× zstd ratio)
- **FP16 for tied embeddings** and last-layer key projections

### Training
- **Muon**: matrix_lr=0.02, WD=0.04, momentum=0.99
- **AdamW**: WD=0.04 for embeddings/scalars
- warmdown=3000, warmup=20, seq_len=2048, batch=786K tokens
- grad_clip=0.3

### Evaluation
- **Sliding window**: stride=64
- **SWA**: start_frac=0.4, every=50 steps (last 40%, 24 checkpoints)

### Ablation (from their PR162 base)
| Change | val_bpb | Delta |
|---|---|---|
| 9L int6 (PR162 base) | 1.1485 | baseline |
| + int5 MLP + 10th layer | 1.1453 | −0.003 |
| + WD=0.04 + warmdown=3000 | 1.1452 | −0.0001 |
| + SWA_start_frac=0.4 | 1.1446 | −0.0006 |
| + bigram=8192 | 1.1434 | −0.0012 |
| + bigram=10240 | **1.1426** | **−0.0008** |

---

## 6. What HAS NOT Been Tried Yet (Opportunity Space)

Based on thorough analysis, all these are **unexplored or underdeveloped** paths:

### 6.1 Architecture Innovations
- **Depth Recurrence with Shared Weights (done right)** — Layer recurrence was tried naively and failed (−0.051), but *progressive stacking* or *parameter sharing with separate norms* could work
- **Mixture of Experts (MoE)** — Route tokens to different expert MLPs. Could dramatically increase capacity within parameter budget
- **Multi-scale/Hierarchical Attention** — Different heads attend at different granularities  
- **Local + Global Attention Patterns** — Mix sliding window attention with sparse global attention
- **State Space Models (Mamba-style)** — SSMs are much more parameter efficient for long sequences
- **RWKV-style Linear Attention** — O(1) per-token inference, very parameter efficient
- **Hyena / H3 Operators** — Sub-quadratic attention alternatives
- **Byte-level models** — Skip the tokenizer entirely (since metric is bits-per-byte)

### 6.2 Compression Innovations
- **Int4 / Int3 Quantization** — More aggressive MLP quantization (int5 already works for MLP)
- **Vector Quantization / Product Quantization** — Cluster weight vectors, store codebook indices
- **Weight Sharing / Parameter Tying** — Share weights across layers (not full recurrence)
- **Low-Rank Factorization** — Decompose weight matrices into UV products
- **Structured Pruning** — Remove entire attention heads or MLP neurons
- **Knowledge Distillation** — Train a larger teacher offline, distill into the 16MB student
- **Neural Network Compression via Entropy Coding** — Arithmetic coding on quantized weights
- **GPTQ / AWQ-style Quantization** — Better quant calibration methods
- **1-bit / 1.58-bit (BitNet)** — Ternary weights {-1, 0, 1}, potentially huge compression

### 6.3 Training Innovations
- **Progressive Training** — Start small (fewer layers/dim), expand during training
- **Curriculum Learning** — Easy examples first, hard examples later
- **Data Selection / Filtering** — Better training data subsets
- **Synthetic Data Augmentation** — Augment training data with paraphrases
- **Distillation from Pre-trained Model** — Use a larger (pre-trained) model to guide training
- **Multi-resolution Training** — Train at multiple sequence lengths
- **Gradient Checkpointing** — Allow larger models to fit in memory

### 6.4 Evaluation Innovations  
- **Better TTT (Test-Time Training)** — Current LoRA TTT gives −0.004. More aggressive TTT with full adaptation
- **Ensemble at Eval** — Average logits from multiple SWA checkpoints at eval time
- **Adaptive Context Length** — Dynamically adjust eval context per document
- **Speculative Decoding Tricks** — Not directly applicable but eval-time compute tricks

### 6.5 Tokenizer Innovations
- **Larger Vocab Size** — 1024 is very small. 2048 or 4096 could improve bpb, but uses more embedding params
- **Character-level or Byte-level** — Eliminate tokenizer overhead entirely
- **BPE Dropout** — Regularization during training
- **Custom Tokenizer** — Trained specifically on this distribution

---

## 7. Failed/Negative Results Worth Noting

These approaches have been tried by others and **did NOT work**:

| Technique | Result | Why It Failed |
|---|---|---|
| **SwiGLU** | ❌ net negative on 8×GPU | 45% slower per step. Better quality doesn't compensate |
| **Layer Recurrence** | ❌ −0.051 bpb (worst!) | Halves training steps. Catastrophic in time-constrained setting |
| **LZMA Compression** | ❌ | Worse than zlib for int8 weight data |
| **Higher Embed LR (0.08)** | ❌ | Hurt convergence |
| **QAT (early attempt)** | ❌ (marginal) | Overhead not worth it for int8. But int6 QAT is hugely beneficial |
| **Weight Decay (short run)** | ❌ | Negligible at very short training runs. Only helps with longer runs |
| **Depth recurrence ×2** | ❌ | See layer recurrence above |

---

## 8. Critical Insights & Patterns

### Insight 1: The Budget Game
The 16MB limit is the binding constraint. Every technique must be evaluated through the lens of: *"Does this save bytes AND improve quality?"*

**Byte budget breakdown (SOTA)**:
- Code: ~48-58K bytes (~0.3% of budget)
- Model: ~15.4-15.9 MB compressed
- Headroom: ~100-600K bytes

### Insight 2: Quantization Quality > Model Quality
The **quantization gap** was originally the dominant bottleneck:
- Baseline int8 gap: **0.014 bpb** — larger than most improvements!
- Int6 QAT eliminates the gap to **0.000 bpb**
- WD helps quantization by keeping weights smaller/smoother
- Lesson: **optimize for post-quant quality**, not just pre-quant quality

### Insight 3: Sliding Window Eval is Free Money
- **−0.032 bpb** for ~70s of eval time
- Zero training cost, zero artifact cost
- Already universally adopted now

### Insight 4: MLP Width > Depth
- Going from 2× → 3× MLP expansion = **−0.029 bpb** (single biggest win)
- Going from 9 → 10 layers = **−0.003 to −0.005 bpb**
- Wider MLPs are more parameter-efficient than deeper networks at this scale

### Insight 5: Every Millisecond Matters
- 600 seconds ÷ 43ms/step = ~13,950 steps
- 600 seconds ÷ 72ms/step = ~8,300 steps
- Technique cost in step time directly reduces total training tokens seen

### Insight 6: Small Vocab = Huge Opportunity  
- Only 1024 tokens → embeddings are tiny (1024 × 512 = 524K params in fp16 = ~1MB)
- BigramHash exploits this: 10240-bucket table captures pair statistics cheaply
- The small vocab means more tokens per document, which means the model sees more "moves"

---

## 9. 🎯 Our Battle Plan: Approaches to Take

### Priority Order (High → Low Impact, accounting for feasibility)

---

### Phase 1: Establish Baseline + Low-Hanging Fruit (Day 1-2)

**Goal: Reproduce current SOTA (~1.143) and set up iteration infrastructure**

1. **Set up the environment** on RunPod 8×H100 or equivalent
2. **Download data**: `python3 data/cached_challenge_fineweb.py --variant sp1024`
3. **Run the baseline** to verify our setup works
4. **Reproduce current SOTA** by implementing all established techniques:
   - 10 layers, 512 dim, MLP 3×, GQA
   - SmearGate + BigramHash(10240)
   - Orthogonal init
   - Int5 MLP / Int6 attention / FP16 embed
   - Muon WD=0.04, momentum=0.99
   - SWA (start_frac=0.4, every=50)
   - Sliding window eval (stride=64)
   - zstd-22 compression

---

### Phase 2: Quick Wins — Squeeze More from Existing Framework (Day 2-5)

**Goal: Get −0.005 to −0.015 bpb. Target: ~1.128-1.137**

#### 2A. Better Quantization (Expected: −0.002 to −0.005)
- [ ] **Int4 MLP quantization**: If int5 works for MLP, try int4 [-8,7]. This could save ~1MB → fund 11th or 12th layer
- [ ] **GPTQ-style calibration**: Use calibration data to find optimal quantization scales instead of simple percentile clipping
- [ ] **Learned quantization step sizes**: Trainable scale/zero-point per channel
- [ ] **Better compression**: Try brotli, lz4, or train a custom entropy coder for the weight distribution

#### 2B. Expand Architecture within Budget (Expected: −0.003 to −0.008)
- [ ] **11 layers with int5 MLP**: Already shown feasible in aruniyer's submission. Combine with int5 for more headroom
- [ ] **MLP 3.5× or 4×**: If quantization saves enough bytes, push MLP wider
- [ ] **Larger BigramHash**: 10240 → 16384 or 20480 buckets. More buckets = fewer collisions
- [ ] **TrigramHash**: Extend bigram to capture 3-token patterns (current → prev → prev-prev)
- [ ] **Tune seq_len for training**: Currently 2048. Try 1536 or 2560 to find the sweet spot

#### 2C. Training Optimization (Expected: −0.001 to −0.003)
- [ ] **WD sweep**: 0.04 may not be optimal. Try 0.03, 0.05, 0.06
- [ ] **LR schedule**: Alternative schedules (cosine, cosine with restarts)
- [ ] **Gradient accumulation tuning**: Different grad_accum_steps
- [ ] **Better warmup**: Try longer warmup or different warmup shapes
- [ ] **Compile optimizations**: Make step time faster → more training steps

---

### Phase 3: Novel Techniques — Our Differentiators (Day 5-15)

**Goal: Get −0.010 to −0.030 bpb. Target: ~1.098-1.128**

#### 3A. 🌟 Enhanced Test-Time Training (HIGH priority)
Current LoRA TTT gives only −0.004 and isn't in the SOTA. There's huge room:
- [ ] **Full-parameter TTT**: Instead of LoRA rank-8, try rank-16 or rank-32
- [ ] **Multi-step TTT**: Current uses 1 gradient step per chunk. Try 2-5 steps
- [ ] **TTT with better optimizer**: Use adam with momentum instead of single SGD step
- [ ] **Selective TTT**: Only adapt MLP layers or only adapt last N layers
- [ ] **Use sliding window + TTT together**: Score with sliding window, then TTT on scored tokens

> [!IMPORTANT]  
> TTT is an **evaluation-time technique** that doesn't count toward training time. The 10-min eval budget allows for significant TTT compute. This is underexploited.

#### 3B. 🌟 MoE (Mixture of Experts) Architecture (HIGH priority)  
- [ ] **Top-1 or Top-2 routing** with 4-8 experts per layer
- [ ] **Shared experts + routed experts** (DeepSeek-V2 style)
- [ ] The key insight: MoE has more total parameters with fewer *active* parameters per token. Since we care about compressed model size, we need MoE experts to share structure  
- [ ] **Expert grouping**: Use the same set of experts across multiple layers (expert sharing)
- [ ] Experts' weights can be int4/int5 since they see fewer tokens → more robust to aggressive quant

#### 3C. 🌟 Low-Rank Parameter Efficiency (HIGH priority)
- [ ] **LoRA-style training**: Train with low-rank adaptations from an orthogonal init
- [ ] **Kronecker-factored weights**: Decompose each weight matrix as W = A ⊗ B
- [ ] **Shared weight basis**: All layers share a common subspace, with per-layer coefficients
- [ ] **SVD-based compression**: After training, SVD-compress weight matrices keeping top-k singular values

#### 3D. Multi-Token Prediction (MEDIUM priority)
- [ ] Train with **2-token prediction head**: Predict both next token and token+2
- [ ] This gives a "free" 2× gradient signal per forward pass
- [ ] The extra head can be discarded at eval time
- [ ] Research shows multi-token prediction significantly improves small model quality

#### 3E. Better N-gram Features (MEDIUM priority)
- [ ] **Character-level embeddings**: Since metric is bpb, character features matter
- [ ] **Conditional n-gram models**: Use a tiny character-level model as pre-processing
- [ ] **Hash kernel expansion**: More sophisticated hash functions for bigram/trigram tables
- [ ] **Trainable hash**: Learnable hash to minimize collisions on this distribution

---

### Phase 4: Radical Ideas — Moonshots (Day 15+)

**Goal: Potentially −0.030+ bpb if they work**

#### 4A. Hybrid SSM-Transformer
- [ ] Replace some transformer layers with Mamba-2 blocks
- [ ] SSMs are more parameter-efficient for modeling long-range dependencies  
- [ ] Could allow deeper networks in same byte budget

#### 4B. BitNet / Ternary Weights  
- [ ] Train with 1.58-bit ternary weights {-1, 0, +1}
- [ ] Each weight = ~1.58 bits → 10× compression vs fp16
- [ ] Could theoretically fit a **100M+ parameter model** in 16MB
- [ ] Extremely aggressive, but the compression potential is enormous

#### 4C. Progressive Training / Growing Network
- [ ] Start training with 6 layers, grow to 12 layers mid-training
- [ ] Early steps train a small, fast core; late steps expand capacity
- [ ] More efficient use of the 10-minute budget

#### 4D. Knowledge Distillation from External Model
- [ ] Pre-train a large model (unconstrained) offline
- [ ] Use its logits as soft targets for our 16MB model
- [ ] Rules say "tuning hyperparameters offline is fine" — distillation targets are arguably hyperparameters
- [ ] ⚠️ This is in a gray area — need to gauge community/organizer sentiment

#### 4E. Custom Tokenizer with Larger Vocab
- [ ] Train a 2048 or 4096 vocab tokenizer
- [ ] Larger vocab = fewer tokens per document = better bpb if model quality holds
- [ ] Trade embedding size for fewer prediction targets
- [ ] ⚠️ Tokenizer changes are scrutinized heavily — must verify bpb calculation is correct

---

## 10. Recommended Execution Order

```
Priority  | Technique               | Expected BPB | Effort | Risk
──────────|────────────────────────--|─────────────|────────|──────
★★★★★    | Reproduce SOTA          | 1.143       | Low    | Low
★★★★★    | Enhanced TTT            | −0.005-0.015| Medium | Low
★★★★☆    | Int4 MLP Quant          | −0.002-0.005| Medium | Low
★★★★☆    | 11-12L via better quant | −0.003-0.008| Medium | Low
★★★★☆    | TrigramHash / Larger BH | −0.001-0.003| Low    | Low
★★★★☆    | Multi-Token Prediction  | −0.002-0.005| Medium | Medium
★★★☆☆    | MoE Architecture        | −0.005-0.015| High   | Medium
★★★☆☆    | Low-Rank Factorization  | −0.003-0.008| Medium | Medium
★★★☆☆    | Better LR Schedule      | −0.001-0.003| Low    | Low
★★☆☆☆    | Hybrid SSM-Transformer  | −0.005-0.020| High   | High
★★☆☆☆    | BitNet Ternary          | −0.010-0.030| Very High | Very High
★☆☆☆☆    | Custom Tokenizer        | −0.005-0.015| High   | High (scrutiny)
```

---

## 11. Quick Reference: Key Numbers

| Metric | Value |
|---|---|
| SOTA val_bpb | 1.14276 |
| To beat SOTA | ≤ ~1.1378 (0.005 nats improvement) |
| Baseline val_bpb | 1.2244 |
| 16MB cap | 16,000,000 bytes |
| 600s training cap | ~8,300-13,900 steps depending on step time |
| Sliding window free gain | −0.032 bpb |
| Largest single technique | MLP 3× (−0.029 bpb) |
| Training dataset | FineWeb 10B tokens (80 shards) |
| Val dataset | FineWeb first 50k documents |
| Eval budget | 10 minutes (separate from training) |
| Required statistical significance | p < 0.01, typically 3 seeds |

---

## 12. Environment & Infrastructure Notes

### RunPod Setup
- Template: [Parameter Golf Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th)
- SSH access required
- All Python deps pre-installed in image
- 8×H100 SXM costs ~$20/hour

### Data Setup
```bash
# Download data (1024 vocab, full training + validation)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Quick smoke test (1 shard only)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

### Training Command Template
```bash
RUN_ID=our_run \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Submission Requirements
1. `README.md` — explain the submission
2. `submission.json` — metadata (name, GitHub ID, val_bpb)
3. `train.log` — training logs (3 seeds for significance)
4. `train_gpt.py` — complete, runnable script

---

## 13. Open Questions to Investigate

1. **What's the theoretical minimum bpb for a 16MB model on FineWeb?** — Understanding the information-theoretic limit helps gauge how much room is left
2. **How does MoE interact with quantization?** — Expert weights see fewer tokens; does this help or hurt quant?
3. **Can TTT be combined with SWA?** — Average SWA model as base, then TTT at eval time
4. **What's the optimal vocab size for 16MB budget?** — 1024 might not be optimal
5. **Is there a way to use the code bytes more aggressively?** — The ~48KB of code barely uses the 16MB. Could we embed lookup tables or compressed data in the code itself?
6. **How much eval budget is being used?** — Current sliding window = ~70-90s. TTT uses more. Total eval budget is 10 min = 600s. There's room for ~500s of TTT compute!

---

> [!TIP]
> **Our biggest edges will come from: (1) Enhanced TTT using the full eval budget, (2) More aggressive quantization to fund architectural expansion, and (3) Novel architecture components (MoE/SSM) that nobody has tried yet.**

Let's go win this. 🏆
