# 11L MLP3x + Int6 PTQ + Muon Momentum 0.99

**val_bpb: 1.1596** | artifact: 15.3MB | 8xH100 600s

## Key Changes from Previous SOTA (1.1753 BPB)

- **11 transformer layers** (up from 8) with dim=432
- **MLP 3x expansion** (up from 2x) — significantly more capacity
- **Muon momentum 0.99** (up from 0.95) — smoother optimization
- **Decoupled weight decay 0.02** on both Muon and Adam
- **Int6 PTQ** (levels=31) with fixed floor bug + zstd-22 compression
- **fp16 embedding passthrough** — embeddings kept at fp16 instead of quantized

## Reproduction

```bash
modal run run_modal.py \
  --command train-8gpu \
  --vocab-size 4096 \
  --variant sp4096 \
  --num-layers 11 \
  --model-dim 432 \
  --mlp-mult 3 \
  --compression-codec zstd \
  --compression-level 22 \
  --fp16-embed 1 \
  --qat-start-frac 1.1 \
  --quant-levels 31 \
  --matrix-lr 0.04 \
  --scalar-lr 0.04 \
  --muon-momentum 0.99 \
  --muon-wd 0.02 \
  --adam-wd 0.02 \
  --warmdown-iters 3000 \
  --train-seq-len 1024 \
  --eval-seq-len 1024 \
  --eval-batch-seqs 256 \
  --max-seconds 600
```

## Architecture

- SP-4096 tokenizer (0.30 tokens/byte)
- 11 layers, dim=432, 8 heads / 4 KV heads (GQA)
- MLP 3x expansion with relu^2 activation
- RoPE positional encoding, encoder-decoder skip connections
- Tied embeddings with overtone spectral init
- 20.3M parameters → 15.3MB artifact (int6+zstd-22)
