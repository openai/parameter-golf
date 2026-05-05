# Faithful mHC-lite

This is a non-record methodological submission exploring manifold-constrained hyper-connections via a faithful standalone `MHCLite` branch.

## Why This Is Interesting

- Uses explicit residual stream expansion and reduction rather than a standard single-stream transformer
- Wraps both attention and MLP branches with `MHCLite`
- Preserves the core hyper-connection idea in a compact GPT-style setting
- Showed clear standalone local signs of life before cloud evaluation

## Method Summary

The submission replaces the ordinary residual path with a multi-stream residual transport mechanism:

1. Expand the token representation into multiple residual streams
2. Apply hyper-connected attention and MLP branches
3. Redistribute information across streams after each branch
4. Reduce back to the base representation for output/logit computation

This tests whether constrained multi-stream routing can improve a small parameter-limited language model.

## Real 8xH100 Results

- Track: non-record 16MB

| Seed | `final_int8_zlib_roundtrip_exact val_loss` | `final_int8_zlib_roundtrip_exact val_bpb` | Total size |
|---|---:|---:|---:|
| `42` | `2.85774078` | `1.69236985` | `2,290,235` bytes |
| `1337` | `2.81310075` | `1.66593378` | `2,317,237` bytes |
| `2024` | `2.81310075` | `1.66593378` | `2,300,762` bytes |

- Mean cloud `val_bpb` across 3 seeds: `1.67474580`

## Strongest Local Signs Of Life

Before cloud testing, the best standalone local branch reached:

- `2.65898106`
- `2.68564709`

on longer proxy runs across two seeds.

## Interpretation

The method is stable, trainable, and genuinely behaves like a viable standalone branch. The 3-seed 8xH100 verification confirms that it transfers reproducibly, but not competitively enough to beat the dense frontier. It is still a worthwhile hyper-connection implementation and negative/partial result submission.

## Bundled Dependency

This folder includes a local copy of `hyper_conn/mhc_lite.py` so the script can run from inside the record folder without depending on the repo-level `implementations/` tree.

## Run Command

```bash
RUN_ID=screen_mhc_seed42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=4 \
TRAIN_BATCH_TOKENS=8192 TRAIN_SEQ_LEN=256 \
VAL_BATCH_SIZE=32768 VAL_MAX_TOKENS=262144 \
VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=200 \
ITERATIONS=200000 MAX_WALLCLOCK_SECONDS=600 \
DISABLE_COMPILE=1 SWA_ENABLED=0 \
TIED_EMBED_LR=0.01 EMBED_LR=0.01 \
MATRIX_LR=0.01 SCALAR_LR=0.01 \
MHC_NUM_STREAMS=4 MHC_NUM_FRACS=1 MHC_DROPOUT=0.0 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Compliance

- [x] Artifact under 16MB
- [x] Runs on real 8xH100 SXM
- [x] No validation leakage
- [x] Non-record submission
