# Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Muon WD + SWA

## Score: mean val_bpb = 1.1458 (3 seeds: 1.1460, 1.1466, 1.1449)

Trained on 8×H100 SXM in 600 seconds. 15.86MB artifact (int6+zstd-22).

## Approach

Seven techniques stacked on the baseline 9-layer, 512-dim GPT:

### 1. Per-Row Int6 Quantization + zstd-22 Compression
MLP and attention weight matrices quantized to int6 ([-32, 31]) with per-row scaling. Tied embeddings remain in fp16 (quantization-sensitive). The last transformer layer's key projection is kept in fp16 to reduce the quantization penalty on late-layer attention. zstd at level 22 provides ~5% better compression than zlib-9 on int6 data.

### 2. 3× MLP Expansion
MLP hidden dimension increased from 1024 (2×) to 1536 (3×), enabled by the byte savings from int6 quantization. This is the single largest contributor to the improvement.

### 3. SmearGate
A learned gate blending each token's embedding with the previous token's embedding, providing lightweight bigram-level context at the embedding layer. Adds ~512 parameters.

### 4. BigramHash Embedding
A 4096-bucket hash table (dim=128, projected to 512) mapping adjacent token pairs to learned embeddings via `(prev_token * 31 + curr_token) % 4096`. Adds ~524K parameters. Complements SmearGate with an additive bigram signal.

### 5. Orthogonal Weight Initialization
All large weight matrices initialized with `orthogonal_(gain=1.0)`. Output projections scaled by `1/sqrt(2 * num_layers)` following muP conventions. Accelerates early convergence.

### 6. Muon Optimizer with Weight Decay
Muon with decoupled weight decay WD=0.04 (swept from 0.01–0.05, optimal at 0.04). Momentum warmup from 0.92 to 0.99 over 1500 steps. AdamW WD=0.01 for embedding and scalar parameters. Weight decay regularizes magnitudes, directly improving int6 quantization quality.

### 7. Stochastic Weight Averaging (SWA)
SWA every 50 steps over the last 50% of training (~30 checkpoints averaged). Produces smoother weight distributions that quantize better. Swept swa_every from 200 down to 25; optimal at 50.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_layers | 9 |
| model_dim | 512 |
| mlp_mult | 3.0 (hidden=1536) |
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 |
| warmdown_iters | 3000 |
| matrix_lr | 0.02 |
| scalar_lr | 0.02 |
| tied_embed_lr | 0.03 |
| muon_momentum | 0.99 (warmup from 0.92 over 1500 steps) |
| muon_weight_decay | 0.04 |
| adamw_weight_decay | 0.01 |
| grad_clip_norm | 0.3 |
| eval_stride | 64 |
| swa_every | 50 |
| swa_start_frac | 0.5 |
| bigram_vocab_size | 4096 |
| bigram_dim | 128 |
| compressor | zstd (level 22) |

## Key Metrics

- **Mean val_bpb: 1.1458** (std: 0.0008)
- Pre-quant val_bpb: 1.1616
- Quantization penalty: 0.016 bpb (int6 vs fp16)
- Training: 7,379 steps in 600s (81.3 ms/step)
- Model params: ~22M
- Artifact size: 15.86MB (int6+zstd-22)

## Reproducibility

Three independent training runs with different random seeds:

| Seed | val_loss | val_bpb |
|------|----------|---------|
| 1337 | 1.93492 | 1.14597 |
| 42 | 1.93591 | 1.14656 |
| 7 | 1.93314 | 1.14492 |
| **Mean** | **1.93466** | **1.14582** |
| **Std** | **0.00139** | **0.00082** |

Improvement over current SOTA (1.1748): **-0.0290 bpb / -0.0503 nats** (p < 0.001).
