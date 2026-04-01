## Progressive Depth + Hedge Mixer

val_bpb = **1.1454** (Hedge Mixer eval, int8+zstd22 roundtrip model)
val_bpb = 1.1966 (sliding window only)
val_bpb = 1.2304 (standard roundtrip)

### Hedge Mixer: 5-Expert Online Ensemble

Eval-time improvement via online mixture of 5 experts using the Hedge algorithm (multiplicative weights). No training data access — n-gram tables built from already-scored tokens only.

| Expert | Source | Role |
|--------|--------|------|
| Neural | Model softmax output | Primary prediction |
| Unigram | Token frequency from scored data | Frequency prior |
| Bigram | P(next\|prev) from scored data | Local context |
| Trigram | Hash table (64K buckets) from scored data | Extended context |
| Entropy | Model confidence weighting | Calibration |

Weights initialized with neural bias (log_weight=2.0), updated via `log_w -= eta * expert_mean_loss` after each batch. The mixer is cold-started (uses pure neural output until 10K tokens scored), then progressively improves as n-gram statistics accumulate.

**Impact: -0.051 bpb** over sliding window eval (1.1966 → 1.1454). This is larger than all architectural improvements combined.

Eval time: 579s on 8xH100 (sequential processing required for n-gram table consistency).

### Architecture (unchanged from PR #835)

3 shared transformer blocks with depth recurrence, progressive depth scheduling unique to shared-weight recurrence.

- **Progressive Depth Training**: Phase 1 (0-40%): 2 repeats ~75ms/step. Phase 2 (40-65%): 3 repeats ~86ms/step. Phase 3 (65-100%): 4 repeats ~96ms/step. 5673 steps in 600s.
- **Cross-Repeat Skip** (#148, Novel): Stateful recurrence — each block receives weighted residual from previous repeat.
- **XSA**: Exclusive Self-Attention on last 4 effective layers.
- **LeakyReLU(0.5)²**: Better gradient flow through 4-repeat recurrence.
- dim=832, 8 heads, 4 KV heads (GQA), MLP 2×, tied embeddings, SWA (18 checkpoints).
- 17.14M params, 15.88MB artifact (int8+zstd22).

### Tuned Hyperparameters

MATRIX_LR=0.018, SCALAR_LR=0.018, TIED_EMBED_LR=0.021, WARMDOWN_ITERS=2000.

Higher LR compensates for progressive depth's shallow early phases. Shorter warmdown gives full LR at full-depth entry.

### Ablation Trajectory

| Change | val_bpb | Delta |
|--------|---------|-------|
| OpenAI Naive Baseline | 1.2244 | — |
| Depth Recurrence 3×4 + Cross-Repeat Skip (#148) | 1.2213 | -0.003 |
| + XSA + LeakyReLU² (#784) | 1.2069 | -0.014 |
| + Progressive Depth (#835) | 1.1980 | -0.009 |
| + LR/Warmdown tuning | 1.1960 | -0.002 |
| + Hedge Mixer (eval) | 1.1454 | -0.051 |
| **Total** | **1.1454** | **-0.079** |

### Command

```
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Credits

Hedge Mixer algorithm adapted from PR #688 (@RoyiRa) and PR #745 (@stukenov).
