# Implementation Plan — Ternary Reasoner

## Core Thesis

**Low-bit inference + structured semantic state + backward semantic correction.**

The Ternary Reasoner is an iterative hierarchical correction architecture where:
1. A ternary-quantized U-Net encoder produces skip-connected representations
2. Recurrent capsule state carriers compress global structure into persistent slots
3. Hadamard-gated feedback adapters apply backward semantic flow — later-layer
   summaries correct earlier representations via element-wise multiplicative modulation
4. Multiple decoder passes iterate toward convergence, each informed by the previous

This is not a standard transformer. It is structured iterative refinement with backward flow.

## Architecture Summary

```
Token Embedding (factorized: embed_dim → model_dim via QATLinear)
    │ + [Optional] BigramHash local context injection
    │
    ├─ Encoder (num_layers // 2 blocks) — runs ONCE
    │    └─ Each block: RMSNorm(+LN damping) → TernaryAttn(+PartialRoPE, +VRL) → RMSNorm → TernaryMLP
    │    └─ Skip connections collected per layer
    │    └─ VRL: first-layer values residually blended into deep-layer attention
    │
    ├─ Capsule Bank (optional) — at encoder-decoder boundary
    │    └─ Soft-assignment pooling into N capsule prototype slots
    │    └─ Recurrent gate blends with previous iteration's capsule state
    │    └─ Gated residual injection back into token stream
    │
    ├─ ITERATIVE CORRECTION LOOP: repeat (feedback_passes + 1) times
    │    │
    │    ├─ Pass 0: blind decoder (no sketch, no capsule history)
    │    │
    │    ├─ Pass 1..N: backward semantic flow
    │    │    └─ Pool previous decoder output into compressed FP8 sketch
    │    │    └─ Update capsule state (recurrent accumulation)
    │    │    └─ Decoder pass with Hadamard-gated correction per block:
    │    │         x *= 1 + gate_m * tanh(proj_m(sketch))   [feature modulation]
    │    │         x += gate_a * proj_a(sketch)              [content injection]
    │    │    └─ U-Net skip connections from encoder (fixed)
    │    │
    │    └─ Early exit if feedback disabled
    │
    ├─ Final RMSNorm → Logit head (tied or untied)
    │
    └─ [Optional] N-gram cache with entropy-adaptive mixing at eval time
```

### Key design choices:
- **Ternary trunk**: all large 2D weight matrices use STE ternary quantization with per-group absmax scaling
- **FP8 islands**: embedding, head, feedback adapters, capsule projections use FP8/FP4 QAT
- **U-Net skip connections**: encoder layers feed decoder layers in reverse order
- **Iterative correction**: N decoder passes, each corrected by backward sketch from previous pass
- **Hadamard-gated adapters**: element-wise multiplicative + additive correction, zero-init (identity at start)
- **Recurrent capsules**: structured semantic state that persists and accumulates across correction iterations
- **Shared block recurrence** (optional): N unique blocks tiled round-robin across depth, per-layer scales still unique
- **BigramHash** (optional): deterministic bigram hashing for cheap local context injection
- **Partial RoPE** (optional): only N dims per head receive rotary encoding, rest attend without position
- **Value Residual Learning** (optional): first-layer attention values blended into deep layers
- **LN Scale Damping** (optional): `1/sqrt(layer_idx+1)` scaling on normalized activations
- **EMA** (optional): exponential moving average of weights, applied before export
- **GPTQ-lite** (optional): per-row clip percentile search before ternary quantization
- **N-gram eval cache** (optional): entropy-adaptive mixing of n-gram empirical probs with neural logits
- **Legal TTT** (optional): adapter-only test-time training on already-scored validation tokens

## Changed Files

| File | What changed |
|------|-------------|
| `train_gpt.py` | Core model with all features |
| `run_cuda_feedback.sh` | Launch script with all env var defaults |
| `setup.sh` | Environment setup (conda + deps) |
| `README.md` | Submission overview and usage |
| `submission.json` | Metadata (placeholder metrics until CUDA runs) |
| `IMPLEMENTATION_PLAN.md` | This file |
| `ABLATION_TODO.md` | Experiment checklist |

## Config Knobs

### Core architecture
| Env var | Default | Description |
|---------|---------|-------------|
| `NUM_LAYERS` | 12 | Total effective layers (encoder + decoder) |
| `MODEL_DIM` | 768 | Hidden dimension |
| `NUM_HEADS` | 8 | Attention heads |
| `NUM_KV_HEADS` | 4 | KV heads (GQA) |
| `MLP_MULT` | 4 | MLP expansion factor |
| `EMBED_DIM` | 254 | Factorized embedding dimension |
| `ACTIVATION` | lrelu2 | relu2, lrelu2, swiglu, relu |
| `BITNET_GROUP_SIZE` | 128 | Ternary quantization group size |
| `XSA_START_LAYER` | 8 | XSA from this layer (-1=off) |
| `LOGIT_SOFTCAP` | 30 | Logit softcapping value |
| `GRAD_CLIP_NORM` | 0.3 | Gradient clipping norm (0=off) |

### Recurrence & sharing
| Env var | Default | Description |
|---------|---------|-------------|
| `SHARED_BLOCKS` | 0 | 0=all unique, 2-3=shared block count tiled across depth |
| `TRAINING_DEPTH_RECURRENCE` | 0 | Extra recurrent passes per block position |
| `EVAL_DEPTH_RECURRENCE` | 0 | Extra recurrent passes at eval time |

### Capsule bank (enabled by default)
| Env var | Default | Description |
|---------|---------|-------------|
| `CAPSULE_ENABLED` | 1 | Enable capsule bank |
| `CAPSULE_NUM` | 16 | Number of capsule slots |
| `CAPSULE_DIM` | 64 | Capsule projection dimension |

### Feedback adapters (backward semantic flow)
| Env var | Default | Description |
|---------|---------|-------------|
| `FEEDBACK_ENABLED` | 1 | Enable feedback decoder replay |
| `FEEDBACK_DIM` | 64 | Feedback sketch dimension |
| `FEEDBACK_SKETCH_TOKENS` | 4 | Pooled sketch token count |
| `FEEDBACK_PASSES` | 1 | Correction passes during training |
| `EVAL_FEEDBACK_PASSES` | 2 | Extra correction passes at eval time |
| `FEEDBACK_REPLAY` | decoder | Replay mode |
| `FEEDBACK_FP_STORAGE` | FP8 | Precision for feedback weights |

### Proven features (all enabled by default)
| Env var | Default | Description |
|---------|---------|-------------|
| `BIGRAM_HASH_ENABLED` | 1 | BigramHash local context (-0.002 BPB) |
| `BIGRAM_HASH_BUCKETS` | 4096 | Hash table size |
| `BIGRAM_HASH_DIM` | 128 | Hash embedding dimension |
| `VRL_ENABLED` | 1 | Value Residual Learning |
| `VRL_START_LAYER` | 10 | First layer to apply VRL |
| `LN_SCALE_DAMPING` | 1 | `1/√(layer+1)` LN scaling |
| `PARTIAL_ROPE_DIMS` | 16 | Dims for RoPE (frees 75% head capacity) |
| `EMA_ENABLED` | 1 | EMA weight averaging (-0.006 BPB) |
| `EMA_DECAY` | 0.997 | EMA decay rate |
| `EMA_START_FRACTION` | 0.5 | When to start EMA |
| `GPTQ_LITE_ENABLED` | 1 | GPTQ-lite clip search (-0.0006 BPB) |
| `GPTQ_LITE_PERCENTILES` | 5 | Number of clip percentiles to search |
| `NGRAM_CACHE_ENABLED` | 0 | Enable N-gram eval cache |
| `NGRAM_MAX_ORDER` | 5 | Maximum n-gram order |
| `NGRAM_ALPHA_BASE` | 0.05 | Base mixing weight |
| `NGRAM_ALPHA_SCALE` | 0.55 | Entropy-adaptive scale |

### TTT
| Env var | Default | Description |
|---------|---------|-------------|
| `TTT_ENABLED` | 0 | Enable test-time training |
| `TTT_SCOPE` | feedback | Which params to adapt |
| `TTT_LR` | 0.002 | TTT learning rate |
| `TTT_EPOCHS` | 1 | TTT epochs per chunk |
| `TTT_CHUNK_TOKENS` | 32768 | Chunk size for TTT |

### Evaluation
| Env var | Default | Description |
|---------|---------|-------------|
| `SLIDING_EVAL` | 1 | Enable sliding window evaluation |
| `SLIDING_EVAL_STRIDE` | 64 | Stride for sliding eval |
| `TEMP_SCALING` | 1 | Enable temperature search |

### Training optimization
| Env var | Default | Description |
|---------|---------|-------------|
| `TRAIN_BATCH_TOKENS` | 786432 | Tokens per training step |
| `TRAIN_SEQ_LEN` | 2048 | Training sequence length |
| `MATRIX_LR` | 0.025 | Muon learning rate |
| `TIED_EMBED_LR` | 0.035 | Tied embedding learning rate |
| `SCALAR_LR` | 0.025 | Scalar parameter learning rate |
| `MUON_MOMENTUM` | 0.95 | Muon final momentum |
| `MUON_WD` | 0.04 | Muon weight decay |
| `ADAM_WD` | 0.04 | Adam weight decay |
| `WARMDOWN_FRACTION` | 0.5 | Time-based warmdown (50% of wallclock) |
| `GRAD_CLIP_NORM` | 0.3 | Gradient clipping norm |

## Status

### Done
- [x] Ternary trunk with STE quantization
- [x] U-Net encoder-decoder with skip connections
- [x] Factorized tied embeddings (embed_dim=254)
- [x] FP8/FP4 QAT for islands
- [x] **Iterative correction loop** (FEEDBACK_PASSES + 1 decoder passes)
- [x] **Hadamard-gated feedback adapters** (multiplicative + additive correction, zero-init)
- [x] **Recurrent capsule bank** with persistent state across correction iterations
- [x] Feedback pooler + FP8 semantic sketch compression
- [x] Shared block recurrence (behind SHARED_BLOCKS flag)
- [x] BigramHash local context injection (behind BIGRAM_HASH_ENABLED flag)
- [x] Partial RoPE (behind PARTIAL_ROPE_DIMS flag)
- [x] Value Residual Learning (behind VRL_ENABLED flag)
- [x] LN Scale Damping (behind LN_SCALE_DAMPING flag)
- [x] EMA weight averaging (behind EMA_ENABLED flag)
- [x] GPTQ-lite clip search (behind GPTQ_LITE_ENABLED flag)
- [x] N-gram eval cache with entropy-adaptive mixing (behind NGRAM_CACHE_ENABLED flag)
- [x] Legal score-first TTT (adapter-only)
- [x] Sliding window evaluation
- [x] Temperature scaling
- [x] Ternary packing (base-3 + bitmask, auto-selects smaller)
- [x] LZMA-9 compression for artifact
- [x] Config flags for every major feature
- [x] All features compile cleanly (py_compile passes)
- [x] Bug review: DDP find_unused fix for shared_blocks
- [x] Bug review: z-loss excluded during eval
- [x] Bug review: GPTQ-lite padding for non-divisible dims
- [x] Bug review: CapsuleBank tuple handling verified
- [x] **XSA (Exclusive Self-Attention)** on last 4 layers — zero params, -0.015 BPB
- [x] **Gradient clipping** (was defined but never applied in training loop)
- [x] **Hyperparameter optimization**: LR, momentum, WD, batch, seq_len, warmdown tuned to match competition-proven values
- [x] **Model depth increase**: 10L → 12L (fills ternary parameter budget)
- [x] **All proven features enabled by default**: lrelu2, BigramHash, VRL, PartialRoPE, LN damping, EMA, GPTQ-lite, XSA
- [x] **Time-based warmdown** (50% of wallclock, not step-based)
- [x] **Eval-time extra feedback passes**: train with 1, eval with 2

### Remaining
- [ ] CUDA training runs to validate all code paths
- [ ] Measure actual artifact sizes per configuration
- [ ] Fine-tune hyperparameters on hardware
- [ ] Measure BPB for each ablation configuration
- [ ] Submission with real metrics

## Artifact Size Budget

- Code: ~100KB
- Ternary trunk (12L, 768d, 4x MLP): ~87M ternary params → ~12MB compressed
- FP8 embeddings (8192 × 254): ~2MB → ~1.5MB compressed
- Feedback adapters: ~100K FP8 params → negligible
- Capsule bank: ~130K FP8 params → negligible
- BigramHash: 4096×128 + 128×768 = ~622K FP8 params → ~0.6MB
- **Budget ceiling**: 16,000,000 bytes total

## Runtime Budget (10 minutes = 600s)

- Training: ~599s (MAX_WALLCLOCK_SECONDS)
- Serialization + roundtrip eval: ~30s
- Temp scaling: ~15s
- Sliding eval (stride=16): ~60-90s
- TTT (optional): ~60-120s
- N-gram cache (optional): ~30-60s
- **Total eval budget**: ~10 minutes additional (separate from training)

## Insights from Research Reports

### Proven high-impact techniques (incorporated):
1. **Sliding eval stride=16**: ~0.025 BPB improvement over non-sliding
2. **LeakyReLU²**: single-component SOTA contributor, -0.0021 BPB
3. **BigramHash**: ~-0.002 BPB, cheap local context
4. **EMA (decay=0.997)**: beats SWA by 0.003 BPB
5. **GPTQ-lite clip search**: zero-cost -0.0006 BPB
6. **Partial RoPE (16/64 dims)**: frees 75% capacity for content
7. **N-gram cache**: -0.07 to -0.16 BPB (the sub-1.0 frontier)
8. **VRL**: best non-TTT result 1.1175 BPB

### Known pitfalls to avoid:
- EMA may conflict with ternary weight quantization — applied to latent FP32 weights only
- torch.compile can dead-code-eliminate QAT paths — verify activation
- N-gram cache is slow (sequential); keep order ≤ 5 for reasonable eval time
- Depth recurrence amplifies ternary quantization error — use sparingly
