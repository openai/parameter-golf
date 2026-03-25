# Vocab 4096 + MLP 3x + Sliding Window Eval

**mean val_bpb: 1.1642** across 3 seeds (1.1650, 1.1640, 1.1637) | **Artifact: ~15.85 MB** (under 16MB)

## Summary

Six improvements stacked on the baseline 9-layer GPT:

1. **Vocab 4096** (up from 1024) — custom SentencePiece BPE tokenizer. Larger vocab means more bytes per token, fewer predictions per byte, directly improving BPB.

2. **3x MLP expansion** (hidden=1536, up from 1024) — enabled by int6 quantization savings. Wider feedforward provides better per-token modeling.

3. **Int6 per-row quantization with STE** — fake int6 quantization during training via Straight-Through Estimator. Model learns weight distributions that survive post-training quantization. Quant gap: +0.005 BPB.

4. **Seq4096 training** — 4x longer context per sequence than the baseline's 1024.

5. **SWA (Stochastic Weight Averaging)** — average of 7 checkpoints during warmdown phase.

6. **Sliding window evaluation** (stride=256, seq_len=4096) — each scored token gets 3840+ tokens of context. Eval time: 148s on 8xH100.

## Configuration

```
VOCAB_SIZE=4096 NUM_LAYERS=8 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
MLP_MULT=3 TIE_EMBEDDINGS=1
TRAIN_SEQ_LEN=4096 EVAL_STRIDE=256
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000
WEIGHT_QUANTIZATION_BITS=6 EMBED_QUANTIZATION_BITS=8
SWA_ENABLED=1
MAX_WALLCLOCK_SECONDS=600
```

## Command

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=v4096_mlp3x \
VOCAB_SIZE=4096 NUM_LAYERS=8 TRAIN_SEQ_LEN=4096 MLP_MULT=3 \
WARMDOWN_ITERS=3000 WEIGHT_QUANTIZATION_BITS=6 EMBED_QUANTIZATION_BITS=8 \
EVAL_STRIDE=256 SWA_ENABLED=1 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
MAX_WALLCLOCK_SECONDS=600 \
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

| Metric | Value |
|--------|-------|
| Steps (10 min cap) | 8,984 |
| Step time | 66.8 ms |
| Model params | 20,994,112 |
| Pre-quant val_bpb | 1.1603 |
| Post-quant sliding window val_bpb | **1.1655** |
| Quantization gap | +0.005 BPB |
| Artifact size | 15,846,785 bytes |
| Eval time (sliding window) | 148s |
| Peak GPU memory | 10,571 MiB |

## 3-Seed Validation

| Seed | val_bpb | Artifact |
|------|---------|----------|
| 1337 | 1.1650 | 15,846,785 bytes |
| 42 | 1.1640 | 15,846,550 bytes |
| 7 | 1.1637 | 15,846,550 bytes |

**Mean: 1.1642, Std: 0.0007**

One-sample t-test against baseline (1.2244): t=-157.3, **p < 0.0001**

## Tokenizer

Custom SentencePiece BPE tokenizer with 4096 vocab, trained on FineWeb. Included as `fineweb_4096_bpe.model`. Tokenizer and pre-tokenized dataset available at [sproos/parameter-golf-tokenizers](https://huggingface.co/sproos/parameter-golf-tokenizers).

## Included Files

- `train_gpt.py` — self-contained training script (1390 lines)
- `train.log` — full training log (seed 1337)
- `train_seed1337.log`, `train_seed42.log`, `train_seed7.log` — 3-seed validation logs
- `submission.json` — leaderboard metadata
- `fineweb_4096_bpe.model` — SentencePiece tokenizer
