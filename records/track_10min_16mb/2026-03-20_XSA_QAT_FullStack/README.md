# Record: Full Stack — Tight SWA + VE + Partial RoPE + LN Scale + XSA4 + Late QAT

Based on PR #374 frontier (1.1246 BPB) with Flash Attention fallback.

## Architecture
- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion with relu-squared
- XSA on last 4 layers (GQA-aware, zero-alloc)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- U-Net skip connections (5 encoder, 6 decoder)
- SmearGate + BigramHash (2048 buckets, dim=128)
- Shared Value Embedding (dim=128, layers 9,10)
- FlashAttention 3 (with FA2 fallback)
- Orthogonal init with proj scaling
- Logit softcap 30.0, tied embeddings

## Training
- Muon: lr=0.025, momentum=0.99, WD=0.04
- AdamW embeddings: lr=0.035, scalars: lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3000 iters (wallclock-based)
- Tight SWA: every 50 steps when scale<0.2
- Late QAT: STE int6 when LR scale<0.1

## Quantization
- Int6 per-row for MLP + attention weights
- Int8 per-row for embeddings
- zstd level 22 compression

## Run
```bash
DATA_PATH=../../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected
- ~6900 steps in 600s
- Sliding window val_bpb: ~1.1246
- Artifact: ~15.7 MB
