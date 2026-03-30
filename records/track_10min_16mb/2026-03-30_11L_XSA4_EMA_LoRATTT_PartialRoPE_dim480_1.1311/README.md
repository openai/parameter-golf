# 11L XSA4 + EMA + LoRA TTT + Partial RoPE + GPTQ-lite (dim480)

**val_bpb: 1.13112** (3-seed mean, std 0.00051) | **~15.5 MB** | 8×H100 SXM Iceland

## Architecture

PR #462 base architecture compressed to fit 16MB with MODEL_DIM=480:

| Component | Setting |
|-----------|---------|
| Layers | 11L |
| MODEL_DIM | 480 |
| NUM_KV_HEADS | 4 |
| MLP_HIDDEN | 1536 (3× MLP) |
| EMA | decay=0.9985 |
| Partial RoPE | 16/64 dims |
| Late QAT | int6 STE at LR scale < 0.15 |
| TTT | Single-pass LoRA (rank=8, lr=0.01, 1 epoch) |
| XSA | 4 deepest layers |
| BigramHash | 8192 buckets, dim=128 |
| SmearGate | Enabled |
| Compression | int6 + zstd-22 (3.83× ratio) |

## 3-Seed Validation (8×H100 SXM, Reykjavík Iceland)

| Seed | Val BPB | Steps | ms/step | Size |
|------|---------|-------|---------|------|
| 1337 | 1.13041826 | 7,847 | 76.47 | 15,489,698 |
| 42 | 1.13161931 | 7,958 | 75.40 | 15,462,345 |
| 7 | 1.13133583 | 7,935 | 75.62 | 15,436,056 |
| **Mean** | **1.13112 (std 0.00051)** | | | |

## Compliance

- [x] Training: ≤600s on 8×H100 SXM
- [x] Artifact: ~15.5MB (under 16,000,000 bytes)
- [x] 3-seed verified

## Reproduce

```bash
MODEL_DIM=480 NUM_KV_HEADS=4 MLP_HIDDEN=1536 NUM_LAYERS=11 \
EMA_ENABLED=1 EMA_DECAY=0.9985 ROPE_DIMS=16 \
LATE_QAT=1 QAT_THRESHOLD=0.15 \
TTT_ENABLED=1 TTT_MODE=lora TTT_LORA_RANK=8 TTT_LORA_LR=0.01 \
TTT_CHUNK_SIZE=256 TTT_EVAL_SEQ_LEN=1024 TTT_BATCH_SIZE=64 \
TTT_EPOCHS=1 TTT_COSINE_DECAY=1 \
XSA_LAYERS=4 BIGRAM_EMBED_DIM=128 \
TRAIN_SEQ_LEN=2048 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
MAX_WALLCLOCK_SECONDS=600 WARMDOWN_ITERS=3500 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Note: Requires Flash Attention 3. On current `runpod/parameter-golf:latest` (PyTorch 2.9.1+cu128), install manually:
```bash
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
```
