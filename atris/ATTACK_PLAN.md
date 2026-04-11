# Parameter Golf — Attack Plan

**Target:** Beat 1.2244 BPB → sub-1.20 BPB → sub-1.18 BPB → absolute minimum
**Constraint:** 16,000,000 bytes (code + int8+zlib model), 10 min on 8xH100 SXM
**Metric:** `final_int8_zlib_roundtrip val_bpb` (lower = better)
**Deadline:** April 30, 2026

---

## Current State

| Entry | BPB | Gap to beat |
|-------|-----|-------------|
| Naive Baseline (10 min) | 1.2244 | — |
| 4-Hour Baseline | 1.2074 | -0.017 |
| **Our target** | **< 1.18** | **-0.044** |

Baseline config: 9 layers, 512 dim, 8 heads, 4 KV heads, 1024 vocab, tied embeddings, Muon optimizer, ~15.86MB artifact.

---

## Attack Vectors (Ordered by Expected Impact)

### A. Architecture — More Capacity Per Parameter

#### A1. Weight Sharing / Depth Recurrence (HIGH IMPACT)
- Share transformer blocks across layers. 3 unique blocks × 3 repeats = 9 effective layers, 1/3 the parameters.
- Universal Transformer style: same block repeated with layer-specific lightweight adapters (scalars/biases only).
- Freed parameters → wider model or more unique blocks.
- **Risk:** Shared weights compress better under zlib (repetitive patterns). Double win.
- **Experiment:** Start with full sharing (1 block × 9), then 3×3, then 2 shared + 1 unique per position.

#### A2. Low-Rank Factorization (MEDIUM IMPACT)
- Factor Q/K/V/O projections: W = UV where U is d×r, V is r×d, r << d.
- Rank 64-128 for a 512-dim model saves significant parameters.
- Can combine with weight sharing for compound savings.
- **Experiment:** Sweep rank from 32 to 256 on attention projections.

#### A3. Sparse MLP / Mixture of Experts (MEDIUM IMPACT)
- Replace single 2x MLP with 4 smaller experts + router.
- More total capacity, same active parameters per token.
- **Risk:** Router overhead, load balancing complexity within 10 min.
- **Experiment:** 2 experts first (simplest), then 4.

#### A4. Sub-Quadratic Attention (LOW IMPACT at 1024 seq len)
- Linear attention, sliding window, etc.
- At seq_len=1024, quadratic attention is fine. Skip unless going longer.

### B. Compression — More Model Per Byte

#### B1. Quantization-Aware Training (HIGH IMPACT)
- Train with fake quantization in the loop. Model learns to be robust to quantization.
- INT8 QAT → INT4 QAT → ternary/binary.
- Current post-hoc INT8 loses ~0.007 BPB (1.2172 → 1.2244). QAT can eliminate this.
- **Experiment:** Add STE (straight-through estimator) for INT8 first, then push to INT4.

#### B2. BitNet / Ternary Weights (HIGH IMPACT)
- 1.58-bit weights {-1, 0, 1}. Massive compression.
- Recent papers show competitive quality at scale.
- Combined with zlib, ternary weights compress extremely well.
- **Experiment:** Replace linear layers with ternary, keep embeddings/norms in higher precision.

#### B3. Structured Pruning + Quantization (MEDIUM IMPACT)
- Train full model, prune channels/heads, then quantize.
- Or train with L1 regularization to encourage sparsity, then prune.

#### B4. Better Compression Algorithm (LOW-MEDIUM IMPACT)
- Replace zlib with zstd (better ratio, same speed) or lzma (best ratio, slower).
- Custom weight encoding: delta coding between layers (especially with weight sharing).
- **Check:** Does the submission format require zlib specifically? → No, just needs to fit in 16MB.

### C. Training Efficiency — More Learning Per Minute

#### C1. Learning Rate / Schedule Optimization (MEDIUM IMPACT)
- Current: linear warmdown. Try cosine, cosine with warm restarts.
- Higher peak LR with aggressive warmdown.
- Per-layer LR scaling.
- **Experiment:** Sweep LR 2x up and 2x down, try cosine schedule.

#### C2. Batch Size / Sequence Length (MEDIUM IMPACT)
- Current: 524K tokens/step, 1024 seq len.
- Larger batch = fewer steps but more stable gradients.
- Shorter sequence (512) = more steps per minute but less context.
- **Experiment:** Try 256K and 1M batch sizes, try 512 and 2048 seq len.

#### C3. Muon Optimizer Tuning (LOW-MEDIUM IMPACT)
- momentum, backend_steps, warmup parameters.
- Newton-Schulz iteration count (currently 5 in backend, 10 in function).
- **Experiment:** Sweep momentum 0.9-0.99, backend_steps 3-7.

#### C4. Data Ordering / Curriculum (LOW IMPACT)
- Sort training data by difficulty (shorter/simpler documents first).
- **Risk:** Fixed shards make this hard without preprocessing.

### D. Evaluation Tricks — Better Score Without Better Model

#### D1. Longer Context at Eval (HIGH IMPACT, LOW EFFORT)
- They explicitly allow eval at any sequence length.
- Train on 1024, eval on 2048 or 4096. More context = better predictions.
- RoPE extrapolation or NTK-aware scaling for longer eval.
- **Experiment:** Just change VAL_BATCH_SIZE eval seq len. Might get 0.01+ BPB for free.

#### D2. Test-Time Training (HIGH IMPACT, COMPLEX)
- Fine-tune on the validation prefix before predicting next tokens.
- Eval budget is 10 min separately from training. That's a LOT of test-time compute.
- **Experiment:** Online SGD on val data during eval pass.

#### D3. Ensembling (MEDIUM IMPACT)
- Train 2-3 models with different seeds, average predictions.
- Must fit ALL models in 16MB → only viable with very small individual models.
- Or: train one model, create pseudo-ensemble via dropout at eval time.

### E. Tokenizer — Different Encoding Efficiency

#### E1. Vocab Size Sweep (MEDIUM IMPACT)
- 1024 is tiny. Each token encodes few bytes.
- 2048 or 4096 vocab: fewer tokens to predict, but larger embedding table.
- BPB is tokenizer-agnostic, so bigger vocab helps IF the model can learn the embeddings.
- **Experiment:** Try 512, 2048, 4096 with appropriate model size adjustments.
- **Risk:** They scrutinize tokenizer changes closely. Must be airtight.

---

## Autoresearch Loop Design

```
┌─────────────────────────────────────────────┐
│                AUTORESEARCH                 │
│                                             │
│  1. Read ATTACK_PLAN.md + past results      │
│  2. Pick highest-impact untested idea       │
│  3. Modify train_gpt.py                     │
│  4. Run: torchrun --nproc_per_node=8        │
│     train_gpt.py (10 min cap)               │
│  5. Read final_int8_zlib_roundtrip val_bpb  │
│  6. If improved ≥ 0.001: KEEP, log result   │
│     If regressed: REVERT, log negative      │
│  7. Repeat                                  │
│                                             │
│  Cost: ~$3.30/experiment (8xH100 @ $20/hr)  │
│  Rate: ~5 experiments/hour                  │
│  Budget: $500 = ~150 experiments            │
└─────────────────────────────────────────────┘
```

---

## Phase Plan

### Phase 1: Foundation (Days 1-3)
- [x] Clone repo, read baseline code
- [x] Map attack vectors
- [ ] Reproduce baseline on 1xH100 (verify ~1.22 BPB)
- [ ] Set up autoresearch harness
- [ ] Apply for compute grant
- [ ] Run MLX smoke tests locally for fast iteration on arch ideas

### Phase 2: Low-Hanging Fruit (Days 4-7)
- [ ] Eval at longer sequence length (D1) — potentially free BPB
- [ ] LR / schedule sweep (C1)
- [ ] Muon hyperparameter sweep (C3)
- [ ] QAT implementation (B1) — eliminate the 0.007 BPB quant loss

### Phase 3: Architecture Innovation (Days 8-14)
- [ ] Weight sharing experiments (A1)
- [ ] Low-rank attention (A2)
- [ ] Vocab size sweep (E1)
- [ ] BitNet/ternary exploration (B2)

### Phase 4: Advanced Techniques (Days 15-25)
- [ ] Test-time training (D2)
- [ ] MoE sparse MLP (A3)
- [ ] Compound improvements — stack all winners
- [ ] Population-based search (ARTEMIS-style) on top performers

### Phase 5: Polish & Submit (Days 26-43)
- [ ] Stack all winning changes
- [ ] Run 5+ seeds for statistical significance
- [ ] Write submission README
- [ ] Submit PR

---

## Key Insights From Our Research

1. **Karpathy's autoresearch** found 20 improvements on a "well-tuned" codebase. The baseline here is explicitly "not SOTA" — there's likely 50+ improvements waiting.

2. **The 5-minute rule transfers.** 10 min fixed budget = identical compute per experiment. Improvements that work here genuinely extract more from same compute.

3. **Weight sharing + quantization = double compression.** Shared weights have identical byte patterns → zlib compresses them to nearly zero. This is the architectural insight most people will miss.

4. **Eval tricks are legal and encouraged.** "We encourage competitors to push the bounds of evaluation methods as aggressively as with training methods." Test-time training with the separate 10-min eval budget is the sleeper weapon.

5. **The scoring gap is small.** 0.005 nats to set a new record. That's achievable with a single good idea.
