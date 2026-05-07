# XSA-7 + BigramHash + ValueResidual + Legal TTT

11-layer parameter-banking GPT trained on 8xH100 SXM via Modal.com under the strict 600s training budget.

## Architecture

- **Layout**: 11 layers, 512 model dim, 1024 vocab (SentencePiece BPE), 8 attention heads, 4 KV heads
- **Attention**: GQA with XSA (cross-sequence attention) on last 7 layers, Partial RoPE (16 dims)
- **Activation**: LeakyReLU(0.5)² with 3.0× MLP expansion
- **Embeddings**: BigramHash (2048 buckets, dim 96) + TrigramHash (1024 buckets, dim 128) + SmearGate
- **Value Embedding**: Token identity reinjection at layers 5, 9, 10 (dim 128)
- **Value Residual**: ResFormer-style value residual connections
- **Tied embeddings**: Input and output embeddings shared

## Optimization

- **Optimizer**: Parallel Muon (batched Newton-Schulz, 5 steps) + AdamW for non-matrix params
- **Parameter Banking**: 3D bank tensors with async reduce-scatter / all-gather overlap
- **EMA**: Decay 0.997
- **SWA**: Stochastic Weight Averaging (every 50 steps)
- **Late QAT**: Quantization-aware training enabled when LR scale < 15%

## Quantization & Compression

- **Format**: int6 per-row quantization + LZMA compression
- **Artifact size**: 15,944,685 bytes (code: 91,881 bytes) — under the 16,000,000-byte cap

## Evaluation

- **Sliding window**: stride=64, seq_len=2048
- **Legal score-first TTT**: SGD (lr=0.002, momentum=0.9), 4 epochs, all blocks unfrozen, 32K-token chunks
- **10+10 compliance**: Training finishes in <600s; evaluation (quant roundtrip + sliding window + TTT) finishes in <600s

## Key Metrics (Best Run — seed 1337)

| Metric | Value |
|--------|-------|
| Training steps | 6,487 |
| Step avg | 90.8 ms |
| Training time | 600,069 ms |
| Post-EMA val_bpb | 1.1406 |
| Sliding window val_bpb (stride=64) | 1.1247 |
| **Legal TTT val_bpb** | **1.1227** |
| Artifact size (int6+lzma) | 15,944,685 bytes |
| Code size | 91,881 bytes |
| Peak GPU memory | 22,859 MiB allocated |

## 3-Seed Statistical Evidence

| Seed | Steps | legal_ttt_val_bpb | final_val_bpb (sliding window) |
|------|-------|-------------------|-------------------------------|
| 1337 | 6,487 | **1.12265** | 1.12468 |
| 2025 | 6,547 | 1.12295 | 1.12514 |
| 27182 | 6,281 | 1.12421 | 1.12616 |
| **Mean** | | **1.12327** | 1.12533 |
| **Std** | | **0.00082** | 0.00075 |

Mean legal_ttt_val_bpb: 1.12327 ± 0.00082

## Infrastructure

- **Hardware**: 8× H100 SXM (Modal.com)
- **Training wallclock**: 600s cap (10-minute budget)
- **Evaluation wallclock**: <600s (sliding window ~90s + TTT ~475s)
- **Flash Attention**: FA3 Hopper kernel via pre-built wheel (flash_attn_3 3.0.0)

## Reproduction

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Environment variables used:
```
VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=96 TRIGRAM_VOCAB_SIZE=1024 TRIGRAM_DIM=128
XSA_LAST_N=7 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=5,9,10 VALUE_RESIDUAL=1
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=4 TTT_FREEZE_BLOCKS=0
MAX_WALLCLOCK_SECONDS=600 ITERATIONS=9000
```
