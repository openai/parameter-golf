# 9x512 SP1024 Baseline — chasewebb

## Score

| Metric | Value |
|--------|-------|
| **val_bpb** | **1.2355** |
| val_loss | 2.0861 |
| Model size (int8+zlib) | 15.87 MB |

## Config

- 9 transformer blocks, width 512
- 8 attention heads, 4 KV heads (GQA)
- MLP expansion 2x
- Vocab size 1024 (SentencePiece BPE)
- Tied embeddings
- Sequence length 1024
- 80 training shards (~8B tokens)
- 8xH100, ~10 min wallclock

## Run Command

```bash
RUN_ID=run_v2 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=580 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Date

2026-03-20
