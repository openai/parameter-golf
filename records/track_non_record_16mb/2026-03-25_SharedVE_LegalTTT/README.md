# Shared ValueEmbedding + Legal TTT

**val_bpb: 1.1201** (3-seed mean, std 0.0002) | **~15.9 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | Artifact |
|------|----------|-------|-------------|-----------------|----------|----------|
| 1337 | 84.9ms | 7,065 | 1.1221 | **1.1199** | -0.0022 | 16,024,522 |
| 42 | 84.9ms | 7,066 | 1.1224 | **1.1202** | -0.0022 | 15,827,158 |
| 2025 | 84.9ms | 7,067 | 1.1225 | **1.1203** | -0.0022 | 15,839,486 |
| **Mean** | **84.9ms** | **7,066** | **1.1223** | **1.1201 (std 0.0002)** | **-0.0022** | |

## Key Innovation: Shared ValueEmbedding

ValueEmbedding (based on ResFormer's value residual idea) reinjects token identity into attention values at deep layers. The standard approach trains a separate `nn.Embedding` table. We instead reuse the existing tied token embedding (`tok_emb`) and learn only a linear projection:

```python
# Standard (separate embedding)
self.ve_shared = ValueEmbedding(vocab_size, ve_dim=128, model_dim=kv_dim)
# 128 * 1024 = 131K extra parameters

# This submission (shared tok_emb)
self.ve_shared = ValueEmbedding(vocab_size, ve_dim=128, model_dim=kv_dim, tok_emb=self.tok_emb)
# Only projection weights + per-layer scales
```

Benefits:
- **Better representations**: tok_emb trains with full gradient signal from the main loss; a separate VE table trains only through the VE path and struggles to learn rich features in limited steps
- **Freed parameter budget**: Eliminating the separate table allows expanding VE from 2 layers (9-10) to 6 layers (5-10), injecting token identity throughout the second half of the network
- **Zero overhead**: No additional forward pass or embedding lookup beyond what tok_emb already computes

## Comparison to Base (PR #549)

| Configuration | Pre-TTT bpb | Post-TTT bpb |
|---|---|---|
| PR #549 (VE128, layers 9-10, separate embed) | 1.1218 | 1.1194 |
| **This (shared tok_emb, layers 5-10)** | **1.1223** | **1.1201** |

Pre-TTT is slightly worse (+0.0005), but the approach is notable for its simplicity and the architectural insight that weight sharing between input embeddings and value embeddings is effective.

## Training Architecture

Built on PR #549 stack with one modification:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3x with LeakyReLU(0.5)² |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| **VE** | **Shared tok_emb, layers 5-10, learned projection + per-layer scales** |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Parameter Banking + Parallel Muon |
| TTT | Legal score-first, 3 epochs, SGD(lr=0.002, momentum=0.9) |

## Run Command

```bash
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

VE_LAYERS defaults to "5,6,7,8,9,10" in the script (line 90).

## Credits

- **Base model + LeakyReLU² + Legal TTT**: PR #549 by @abaybektursun
- **ValueEmbedding / ResFormer**: Qin et al., "Value Residual Learning for Alleviating Attention Concentration in Transformers" (2024)
- **Parameter Banking + Parallel Muon**: PR #399 by @abaybektursun
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
- **Base architecture**: PR #414 by @signalrush
