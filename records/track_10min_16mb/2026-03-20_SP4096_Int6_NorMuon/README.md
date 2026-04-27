# SP4096: Larger Vocabulary + Int6 Quantization for Better BPB

## Summary

**val_bpb = 1.2012** | **Artifact: 14,342,773 bytes** (under 16MB)

Two improvements stacked on the baseline:

1. **SP4096 tokenizer** -- SentencePiece BPE with vocab_size=4096 compresses text 26% more efficiently than the baseline sp1024 (0.306 vs 0.414 tokens/byte). Better compression directly reduces the tokens_per_byte multiplier in BPB.

2. **Int6 quantization + zstd** -- Per-row int6 quantization ([-31,31]) with STE fake quantization during training, fp16 embedding passthrough, and zstd-22 compression. Saves ~3MB vs int8+zlib, fitting the larger vocabulary model under 16MB.

Additional tuning: NorMuon optimizer, halved learning rates (matrix=0.02, scalar=0.02, embed=0.03), extended warmdown (3000 iterations), higher Muon momentum (0.99).

## Key Metrics

| Metric | Value |
|--------|-------|
| **val_bpb (post-quant)** | **1.2012** |
| Pre-quant val_bpb | 1.2012 |
| Artifact size | 14,342,773 bytes |
| Training steps | 11,497 (wallclock-limited) |
| Step avg | 52.1ms |
| Hardware | 8xH100 SXM 80GB (RunPod) |

## Configuration

```
VOCAB_SIZE=4096
NUM_LAYERS=9
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=2
TIE_EMBEDDINGS=1
TRAIN_SEQ_LEN=1024
TRAIN_BATCH_TOKENS=524288
MATRIX_LR=0.02
SCALAR_LR=0.02
TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000
```

## Tokenizer

SentencePiece BPE trained on 500K FineWeb documents with byte_fallback=True, split_digits=True, nmt_nfkc normalization. Tokenizer model included in the submission artifact.

Tokenizer and pre-tokenized dataset available from the author on request.

## Command

```bash
pip install zstandard
NCCL_IB_DISABLE=1 \
VOCAB_SIZE=4096 \
TIE_EMBEDDINGS=1 \
DATA_PATH=./data/export_sp4096/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/export_sp4096/tokenizers/fineweb_4096_bpe.model \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `train_gpt.py` -- training script with int6+zstd quantization, STE QAT, NorMuon
- `submission.json` -- leaderboard metadata
- `README.md` -- this file
- `train.log` -- full training log from 8xH100 run
