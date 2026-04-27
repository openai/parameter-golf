# Hymba: Hybrid Attention + Mamba SSM Language Model

**OpenAI Parameter Golf Challenge submission**
[https://github.com/openai/parameter-golf](https://github.com/openai/parameter-golf)

## Overview

This submission trains a **Hymba** (Hybrid Mamba + Attention) language model on the FineWeb dataset
using a 1024-token SentencePiece vocabulary. The model combines Mamba SSM and grouped-query
attention in parallel within each block, with U-Net skip connections across layers.

**Primary metric:** `val_bpb` — bits per byte on the FineWeb validation split (lower is better).
**Artifact limit:** 16,000,000 bytes (code bytes + compressed model bytes).

---

## Architecture

- **Model:** Hymba — every block runs Attention and Mamba SSM in parallel with a learned merge gate
- **Embedding:** BigramHash (4096 buckets, 128-dim) + optional SmearGate
- **Attention:** Grouped-query attention (8 heads, 4 KV heads) with RoPE / RNoPE alternation
- **SSM:** Mamba-style selective scan (expand=1, conv kernel=3, state=4)
- **MLP:** SwiGLU with learnable per-layer scales
- **Skip connections:** U-Net encoder/decoder skip with learnable weights
- **Residual mixing:** learnable blend of `x` and initial embedding `x0`
- **Quantization:** logit softcap (30.0)
- **Weight averaging:** BEMA (bias-corrected EMA, arxiv 2508.00180)
- **Init:** orthogonal init for weight matrices, zero init for output projections
- **Embeddings:** tied input/output embeddings

**Default config:** `NUM_LAYERS=7, MODEL_DIM=512, NUM_HEADS=8, NUM_KV_HEADS=4, MLP_MULT=4`

---

## Training

- **Optimizer:** Muon (matrix weights) + Adam (embeddings, scalars)
- **Schedule:** cosine warmdown triggered by wallclock time
- **Sequence length curriculum (SLC):** 128 → 1024 over first 2000 steps
- **Budget:** 600 s wall clock (10 min on 8×H100)
- **Batch:** 65,536 tokens/step (global)

---

## Requirements

```
torch>=2.3.0
numpy>=1.26
sentencepiece>=0.2.0
mamba-ssm[causal-conv1d]  # requires CUDA build
huggingface-hub>=1.7.2    # required for HF Hub / Endpoint deployment script
```

Install:

```bash
pip install torch numpy sentencepiece
pip install mamba-ssm[causal-conv1d]
pip install "huggingface-hub>=1.7.2"
```

Dataset (FineWeb with 1024-token vocabulary):

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
```

This downloads data to `./data/datasets/fineweb10B_sp1024/` and the tokenizer to
`./data/tokenizers/fineweb_1024_bpe.model`.

---

## Run Command

```bash
NUM_LAYERS=6 MLP_MULT=3 TIME_BUDGET_SECONDS=600 MTP_HEADS=0 WARMDOWN_ITERS=3000 EMA_DECAY=0.99 MLA_KV_LORA_RANK=128 MLA_QK_ROPE_DIM=32 python train_gpt.py 
```

Multi-GPU (8×H100, competition track):

```bash
RUN_ID=hymba_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=6 MLP_MULT=3 TIME_BUDGET_SECONDS=600 MTP_HEADS=0 WARMDOWN_ITERS=3000 EMA_DECAY=0.99 MLA_KV_LORA_RANK=128 MLA_QK_ROPE_DIM=32 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

