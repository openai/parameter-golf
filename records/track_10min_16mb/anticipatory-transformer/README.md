# Anticipatory Transformer

## Approach

Builds on proven SOTA techniques (10L, Int5-MLP/Int6-Attn mixed quantization, BigramHash 10240, SmearGate, OrthoInit, Muon WD=0.04, SWA, sliding window eval at stride=64, zstd-22) and adds two parameter-free innovations from anticipation geometry theory:

### Innovation 1: Entropy-Weighted Training Loss (parameter-free)

Standard CE loss treats all tokens equally. But language is highly skewed: most tokens are near-deterministic (articles, common words, punctuation) while a minority carry the actual information. We compute per-token entropy from the model's own distribution and use it to upweight hard tokens:

```
weights = 1.0 + alpha * normalized_entropy
loss = mean(per_token_CE * weights / mean(weights))
```

This focuses training capacity on the tokens that actually matter for compression, without changing the model architecture or adding parameters. Alpha=0.15 gives a mild upweighting (easy tokens get weight ~1.0, hardest tokens get ~1.15).

**Why this works**: The model wastes gradient signal on tokens it already predicts well. By upweighting hard tokens, we get more useful gradient per step, equivalent to training on a more informative curriculum without any data changes.

### Innovation 2: Trajectory-Scaled Attention (parameter-free)

In each attention layer, we compute a per-position "commitment" signal from the local stability of hidden states. The signal is derived from the velocity of representation changes between adjacent positions:

```
velocity = ||h[t] - h[t-1]||
commitment = 1 / (1 + strength * cumulative_mean_velocity)
attention_output *= commitment
```

Positions where the hidden representation is stable (low velocity between neighbors) get a slight boost to their attention output. This is the anticipation geometry insight: tokens that have "committed" to a stable representation carry stronger, more reliable signal for downstream predictions.

**Why this works**: It provides a lightweight form of adaptive compute, amplifying the contribution of positions where the model is confident while dampening noisy, unstable representations. No extra parameters needed.

### Base Techniques (from SOTA)

1. **10 layers, 512 dim, 3x MLP** (hidden=1536)
2. **Mixed Int5/Int6 quantization** with per-row scaling + zstd-22
3. **BigramHash embedding** (10240 buckets, dim=128, projected to 512)
4. **SmearGate** (learned adjacent token blending)
5. **Orthogonal initialization** with muP output scaling
6. **Muon optimizer** with WD=0.04, momentum warmup 0.92->0.99 over 1500 steps
7. **SWA** every 50 steps over last 40% of training
8. **Sliding window eval** at stride=64
9. **3% magnitude pruning** before quantization
10. **Gradient clipping** at 0.3

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_layers | 10 |
| model_dim | 512 |
| mlp_mult | 3.0 (hidden=1536) |
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 |
| warmdown_iters | 3000 |
| matrix_lr | 0.02 |
| scalar_lr | 0.02 |
| tied_embed_lr | 0.03 |
| muon_weight_decay | 0.04 |
| grad_clip_norm | 0.3 |
| eval_stride | 64 |
| swa_every | 50 |
| swa_start_frac | 0.4 |
| bigram_vocab_size | 10240 |
| bigram_dim | 128 |
| entropy_loss_alpha | 0.15 |
| trajectory_strength | 0.1 |
| compressor | zstd (level 22) |

## Key Design Decisions

1. **Entropy-weighted loss only during training**: The loss reweighting is disabled during eval (standard CE used for val_bpb calculation), ensuring fair comparison. The improvement comes from better training dynamics, not evaluation tricks.

2. **Trajectory bias is parameter-free**: No extra learned weights. The commitment signal is derived purely from the geometry of the hidden state trajectory. This means zero extra compression cost.

3. **Conservative hyperparameters**: Alpha=0.15 and strength=0.1 are conservative choices. The innovations are designed to be complementary to existing optimizations, not to replace them. If they hurt, they can be disabled via environment variables.

## Usage

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Disable innovations independently:
```bash
ENTROPY_LOSS_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
TRAJECTORY_ATTN_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
