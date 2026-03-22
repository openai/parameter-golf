# DominationV2: 11L XSA4 + EMA + TTT + Int6 MLP3x

**Mean val_bpb: 1.1377** (3 seeds verified) | **15.50 MB** | 8xH100 SXM, 600s train + ~280s eval

## Key Techniques

1. **Exclusive Self Attention (XSA)** on last 4 layers — removes self-value bias from attention output via orthogonal projection. Zero new parameters. Based on arXiv:2603.09078.
2. **EMA** (decay=0.997) replacing SWA — exponential moving average every step for smoother weight averaging.
3. **Test-Time Training (TTT)** — 3-epoch full-model SGD (lr=1e-4) on the already-quantized model at eval time. Only trains on validation tokens that have already been graded.
4. **11 layers**, 512 dim, 8 heads, 4 KV heads (GQA), MLP 3x (hidden=1536).
5. **Per-dimension SmearGate** — 512 independent blend ratios.
6. **BigramHash (2048x128)** — hashed bigram token-pair context.
7. **Int6 per-row quantization** on MLP + attention weights, int8 for embedding, zstd-22.
8. **Muon optimizer** with WD=0.04, momentum 0.99, LR=0.025.
9. **Orthogonal init + muP scaling**, U-Net skip connections.
10. **Sliding window eval** stride=64.

## Results (3 seeds, 8xH100 SXM)

| Seed | val_bpb | Artifact |
|------|---------|----------|
| **1337** | **1.13668** | **15.50 MB** |
| 42 | 1.13727 | 15.50 MB |
| 7 | 1.13926 | 15.50 MB |

**Mean: 1.13774** | Range: 0.00258

## Run command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
TTT_ENABLED=1 TTT_EPOCHS=3 TTT_LR=0.0001 \
MUON_WD=0.04 WEIGHT_DECAY=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
MIXED_QUANT_INT6_CATS=mlp,attn \
TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=2048 \
MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## References

- Exclusive Self Attention: [arXiv:2603.09078](https://arxiv.org/abs/2603.09078)
