# 10L Int5-MLP + SmearGate + BigramHash + Late QAT

Record candidate, ready to run. `submission.json` and `train.log` are placeholders until we have three real 8xH100 SXM runs logged.

## Why this combination

We're pulling together the pieces that have actually worked across the top submissions:

- mixed int5 MLP / int6 attention export, which buys enough artifact budget for a 10th layer
- SmearGate + BigramHash for cheap token-pair context
- orthogonal init with muP-style output projection scaling
- decoupled Muon weight decay at 0.04
- SWA during warmdown
- late QAT kicking in at 85% of wallclock (not always-on STE, which has consistently underperformed)
- sliding-window eval, stride 64, with full-tail handling

The idea is straightforward. Start from the best public 10L mixed-precision stack. Keep the local-context gains that SmearGate runs have over the older int6/MLP3x cluster. And only add the late-stage export-aware training that keeps beating full-run STE in head-to-head comparisons.

## Default recipe

- `NUM_LAYERS=10`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=3.0`
- `TRAIN_BATCH_TOKENS=786432`
- `TRAIN_SEQ_LEN=2048`
- `EVAL_SEQ_LEN=2048`
- `EVAL_STRIDE=64`
- `BIGRAM_VOCAB_SIZE=4096`
- `BIGRAM_DIM=128`
- `MATRIX_LR=0.025`
- `SCALAR_LR=0.02`
- `TIED_EMBED_LR=0.03`
- `MUON_WEIGHT_DECAY=0.04`
- `ADAM_WEIGHT_DECAY=0.01`
- `SWA_ENABLED=1`
- `SWA_START_FRAC=0.5`
- `SWA_EVERY=50`
- `QAT_ENABLED=1`
- `QAT_START_FRAC=0.85`
- `KEEP_LAST_K_FP16=1`
- `REQUIRE_ZSTD=1`

## Before you run

The script expects `zstandard` so the artifact gets zstd-22 compression. Install it first:

```bash
pip install zstandard
```

If you just want a quick smoke test, `REQUIRE_ZSTD=0` falls back to zlib. But that won't match the intended record setup.

## Reproduction

```bash
RUN_ID=10l_int5mlp_smearbigram_lateqat_seed1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 \
pip install zstandard && \
torchrun --standalone --nproc_per_node=8 \
./records/track_10min_16mb/2026-03-20_10L_Int5MLP_SmearBigram_LateQAT/train_gpt.py
```

Three-seed sweep:

```bash
for SEED in 1337 42 7; do
  RUN_ID=10l_int5mlp_smearbigram_lateqat_seed${SEED} \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  SEED=${SEED} \
  torchrun --standalone --nproc_per_node=8 \
  ./records/track_10min_16mb/2026-03-20_10L_Int5MLP_SmearBigram_LateQAT/train_gpt.py
done
```

## What to record after the runs

- `final_mixed_int5int6_roundtrip_exact val_loss:<...> val_bpb:<...>`
- `Serialized model mixed-int5int6+zstd:<...>bytes`
- `Total submission size mixed-int5int6:<...>bytes`
- last train step reached under the 600s cap
- eval wall time

## Files

- `train_gpt.py` is the record candidate script
- `train.log` is a placeholder, replace it with real multi-seed logs
- `submission.json` is the metadata template, fill it in after the runs
