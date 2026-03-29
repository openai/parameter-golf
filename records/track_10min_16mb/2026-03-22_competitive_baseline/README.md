# Competitive Baseline: 10L GQA + Mixed Int6/Int8 + SWA + Seq4096

## Score: mean val_bpb = 1.1536 (3 seeds: 1.1530, 1.1545, 1.1533)

Trained on 8×H100 SXM in 600 seconds. 15.74MB artifact (mixed int6/int8 + zstd).

## Approach

A competitive baseline combining several established techniques from the leaderboard on a 10-layer, 512-dim GPT with Group Query Attention:

### 1. Extended Architecture: 10 Layers with GQA
Increased from the naive baseline's 9 layers to 10 layers, with Group Query Attention using 8 query heads and 4 KV heads. GQA reduces KV cache memory requirements while maintaining attention quality, enabling the extra layer within the 16MB budget.

### 2. 3× MLP Expansion
MLP hidden dimension increased to 1536 (3× model_dim), enabled by byte savings from aggressive quantization. Wider MLPs improve the model's per-layer representational capacity.

### 3. Mixed Quantization (Int6/Int8) + zstandard Compression
Block weight matrices (attention and MLP projections) quantized to int6 ([-32, 31]) with per-row scaling. Token embeddings kept at int8 precision since they are more quantization-sensitive. zstandard compression applied to the serialized model for additional savings. Control tensors (scales, gains, residual mixing weights) kept in FP16.

### 4. Stochastic Weight Averaging (SWA)
SWA applied with ratio 0.4, averaging checkpoints from the last 40% of training. This produces smoother weight distributions that quantize better and improve generalization.

### 5. Extended Sequence Length (4096)
Training with seq_len=4096 provides longer context windows, improving the model's ability to capture longer-range dependencies during training.

### 6. Muon Optimizer with Weight Decay
Muon optimizer (with Newton-Schulz orthogonalization) for 2D matrix parameters with lr=0.04 and weight decay=0.04. AdamW for scalar/embedding parameters. Weight decay regularizes weight magnitudes, directly improving quantization quality.

### 7. Tied Embeddings
Input and output embeddings share weights, saving parameters. Embedding initialized with std=0.005 and uses a dedicated lr=0.6.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_layers | 10 |
| model_dim | 512 |
| mlp_mult | 3.0 (hidden=1536) |
| num_heads | 8 |
| num_kv_heads | 4 (GQA) |
| train_seq_len | 4096 |
| train_batch_tokens | 131,072 |
| warmdown_iters | 1200 |
| embed_lr | 0.6 |
| matrix_lr | 0.04 |
| scalar_lr | 0.04 |
| head_lr | 0.008 |
| muon_momentum | 0.95 |
| muon_weight_decay | 0.04 |
| swa_ratio | 0.4 |
| logit_softcap | 30.0 |
| compressor | zstandard |
| vocab_size | 1024 |
| tie_embeddings | true |

## Key Metrics

- **Mean val_bpb: 1.1536** (std: 0.0009)
- Pre-quant val_bpb: 1.1687
- Quantization penalty: ~0.015 bpb (mixed int6/int8 vs fp16)
- Training: 7,184 steps in 600s (~83.5 ms/step)
- Model params: ~24M
- Artifact size: 15.74MB (mixed int6/int8 + zstd)

## Reproducibility

Three independent training runs with different random seeds:

| Seed | val_loss | val_bpb |
|------|----------|---------|
| 1337 | 1.94685 | 1.15303 |
| 42 | 1.94932 | 1.15450 |
| 7 | 1.94728 | 1.15329 |
| **Mean** | **1.94782** | **1.15361** |
| **Std** | **0.00152** | **0.00090** |

## Notes

This is a non-record submission combining established techniques into a clean competitive baseline. The submission does not beat the current SOTA (1.1428) but demonstrates a solid integration of mixed quantization, extended sequence length, SWA, and GQA within the 16MB constraint.
