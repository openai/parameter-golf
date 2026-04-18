# SFT — Stochastic Feature Transformer (Random Feature Adapters)

## Key Innovation: Learning Adapters on Random Linear Maps

This submission implements **RandomFeatureMLP** — the specific approach OpenAI requested
in "Requests for PRs" (Issue #942): *"Learning adapters on random linear maps"*.

### Core Idea

In a standard Transformer MLP:
```
y = W_out(σ(W_in(x)))  # W_in and W_out are BOTH learned
```

In our RandomFeatureMLP:
```
R = random_gaussian(seed=314159 + layer_idx)  # Fixed, 0 bytes stored
y = W_out(σ(R @ x))                           # Only W_out is learned
```

The random projection matrix `R` is regenerated from a deterministic seed at load time,
costing **exactly 0 bytes** in the artifact. This halves MLP parameter storage per layer.

### Why This Works

- **Random Features Theory** (Rahimi & Recht, NeurIPS 2007): Random projections followed
  by nonlinear activations approximate kernel functions. The learned output projection
  `W_out` acts as an adapter that maps these random features to useful representations.
- **Parameter Efficiency**: With 50% fewer MLP parameters per layer, we can fit more
  layers or use saved space for model quality improvements.
- **Training Speed**: No gradients flow through `R` → backward pass is ~25% cheaper
  per MLP layer, enabling more training steps within the 10-minute budget.

### Architecture

| Component | Detail |
|-----------|--------|
| Tokenizer | SentencePiece BPE 8192 |
| Layers | 11 physical (17 virtual with depth recurrence) |
| Model dim | 512 |
| Heads | 8 query / 4 KV (GQA) |
| Head dim | 64 |
| MLP | RandomFeatureMLP 4× expansion (2048 hidden) |
| RoPE | Partial (16/64 dims), base=10000 |
| QK-Gain | 5.25 (learnable per head) |
| Logit softcap | 30.0 |
| Embeddings | Tied input/output |
| Depth recurrence | Loop layers 3-5, 2 extra iterations, enable at 35% |
| Parallel residuals | From layer 7 (GPT-J style) |
| U-net skip | Sigmoid-gated skip connections |
| LN scale | 1/√(2L+1) residual scaling |

### Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Muon (row-normalized, WD=0.095) + Adam |
| Matrix LR | 0.022 |
| Scalar LR | 0.02 |
| Embed LR | 0.03 (tied) |
| Muon momentum | 0.99 (warmup from 0.92 over 1500 steps) |
| Batch tokens | 786,432 |
| Seq len | 2048 |
| Warmdown | 72% of wallclock |
| EMA | decay=0.9965 |
| Grad clip | 0.3 |

### Quantization & Compression

- **GPTQ int6** for weight matrices (SDClip k=12.85)
- **int8** for embeddings (SDClip k=20.0)
- **Brotli-11** compression (fallback to zlib)

### Evaluation

- **Sliding window**: seq_len=2048, stride=64
- **Score-first TTT**: SGD lr=0.005, momentum=0.9, 3 epochs, 32K token chunks
  (legal per Issue #1017 Condition 3: score-before-update)

### Parameter Budget

| Component | Params (stored) | Params (free) |
|-----------|-----------------|---------------|
| Embedding | 4,194,304 | — |
| Attention (×11) | 8,650,752 | — |
| MLP output proj (×11) | 11,534,336 | — |
| MLP random proj (×11) | — | 11,534,336 |
| Controls/norms | ~50,000 | — |
| **Total** | **~24.4M** | **~11.5M** |

Stored parameters at int6+Brotli ≈ 13.4 MB (2.6 MB headroom under 16 MB cap).

### How to Run

```bash
# Smoke test (1 GPU)
torchrun --nproc_per_node=1 train_gpt.py

# Full run (8×H100 SXM)
torchrun --nproc_per_node=8 train_gpt.py
```
