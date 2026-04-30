# Faithful Conditional Memory

This is a non-record methodological submission exploring a faithful standalone conditional-memory architecture inspired by scalable lookup memory. The model augments a transformer with compressed n-gram lookup tables, contextual key/value projections, and internal-layer memory injection.

## Why This Is Interesting

- Uses tokenizer-normalized compressed lookup keys rather than a plain dense residual path
- Builds prime-sized multi-hash n-gram tables
- Injects learned memory features into selected transformer layers
- Showed strong standalone local signs of life before cloud evaluation

## Method Summary

The submission adds a memory path that:

1. Normalizes tokenizer pieces into compressed text identities
2. Hashes 2-gram and 3-gram contexts into multiple memory heads
3. Looks up learned memory embeddings
4. Projects them into contextual keys and values
5. Gates and injects them back into the transformer at selected layers

This gives the model explicit reusable token-pattern memory in addition to self-attention.

## Real 8xH100 Results

- Track: non-record 16MB

| Seed | `final_int8_zlib_roundtrip_exact val_loss` | `final_int8_zlib_roundtrip_exact val_bpb` | Total size |
|---|---:|---:|---:|
| `42` | `2.72026400` | `1.61095535` | `2,773,498` bytes |
| `1337` | `2.75858221` | `1.63364760` | `2,769,754` bytes |
| `2024` | `2.73756576` | `1.62120154` | `2,772,946` bytes |

- Mean cloud `val_bpb` across 3 seeds: `1.62193483`

## Strongest Local Signs Of Life

Before cloud testing, the best standalone local branch reached:

- `2.55013473`
- `2.55742361`
- `2.60430848`

on longer proxy runs across three seeds.

## Interpretation

This method had one of the strongest heavyweight standalone local signals we found. The 3-seed 8xH100 verification confirms that it transfers reproducibly, but not competitively enough to challenge the dense frontier. It is submitted as a faithful implementation and negative/partial result for future work rather than as a leaderboard contender.

## Run Command

```bash
RUN_ID=screen_memory_seed42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 \
TRAIN_BATCH_TOKENS=8192 TRAIN_SEQ_LEN=256 \
VAL_BATCH_SIZE=32768 VAL_MAX_TOKENS=262144 \
VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=200 \
ITERATIONS=200000 MAX_WALLCLOCK_SECONDS=600 \
DISABLE_COMPILE=1 SWA_ENABLED=0 \
USE_EXCLUSIVE_SELF_ATTENTION=1 \
TIED_EMBED_LR=0.003 EMBED_LR=0.003 HEAD_LR=0.003 \
MATRIX_LR=0.003 SCALAR_LR=0.003 WEIGHT_DECAY=0.01 \
ENGRAM_VOCAB_SIZES=2048,2048 ENGRAM_LAYER_IDS=1,3 \
ENGRAM_N_EMBED_PER_NGRAM=128 ENGRAM_N_HEAD_PER_NGRAM=4 \
ENGRAM_KERNEL_SIZE=4 SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Compliance

- [x] Artifact under 16MB
- [x] Runs on real 8xH100 SXM
- [x] No validation leakage
- [x] Non-record submission
