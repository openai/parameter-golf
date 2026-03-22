# 11L Next-Gen Stack: val_bpb = 1.1399

## Summary

11-layer transformer with the full competitive stack achieving **val_bpb = 1.1399** on sliding window evaluation (stride=64). Artifact: 15.79MB (under 16MB limit).

## Architecture & Techniques

| Component | Details |
|-----------|---------|
| **Layers** | 11 transformer layers, 512 dim, 8 heads, 4 KV heads (GQA) |
| **MLP** | 3x expansion (hidden=1536), ReLU² activation |
| **XSA** | Exclusive Self Attention on last 4 layers (arXiv:2603.09078) |
| **RoPE** | Partial RoPE (16 of 64 dims), NTK-aware base=50000 |
| **LN Scale** | 1/sqrt(layer_idx+1) depth-aware pre-norm scaling |
| **Quantization** | Int5 mixed precision + Late QAT STE (last ~10% of warmdown) |
| **Compression** | zstd-22 + GPTQ-lite clip search (5 candidates per matrix) |
| **SmearGate** | Learned sigmoid token blending gate (~512 params) |
| **BigramHash** | 2048-bucket hash embedding for token-pair features (dim 128) |
| **Initialization** | Orthogonal + muP scaling |
| **Optimizer** | Muon (WD=0.04, momentum=0.99, warmup 0.92→0.99 over 1500 steps) |
| **SWA** | Tight SWA (scale<0.2, ~7 checkpoint average, zero penalty) |
| **Attention** | FlashAttention 3 (Hopper native) |
| **Sequence** | Train@2048, eval@2048 |
| **Eval** | Sliding window stride=64 |

## Results

| Seed | Steps | Step Avg | val_bpb | Artifact |
|------|-------|----------|---------|----------|
| 1337 | 5,660 | 101ms | **1.1399** | 15.79MB |

Training time: 600s (wallclock cap). 8xH100 SXM.

## Reproduction

```bash
RUN_ID=submission \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=0 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Requires `pip install zstandard flash-attn`.
