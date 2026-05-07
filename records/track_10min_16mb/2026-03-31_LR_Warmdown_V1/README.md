# LR Warmdown v1

Warmup and warmdown schedule tuning on top of the official Parameter Golf baseline.

## Summary

This submission modifies the official baseline by changing only:

- `WARMUP_STEPS`
- warmup and warmdown timing

Primary goal:

- improve official `val_bpb` while staying within the official artifact and wallclock limits

Important scope note:

The official `train_gpt.py` baseline uses multiple optimizer LR controls rather than one global LR:

- `EMBED_LR`
- `HEAD_LR`
- `TIED_EMBED_LR`
- `MATRIX_LR`
- `SCALAR_LR`

This submission family does not change those values.

## Why This Should Help

The official challenge is tightly constrained, so small optimization gains can matter a lot.
This run family tests whether a better schedule improves the final model quality without changing architecture, tokenizer, evaluation behavior, or baseline optimizer LR values.

Expected upside:

- cleaner optimization late in training
- low artifact-size risk
- low rule ambiguity

Tradeoff:

- gains may be modest
- results must be confirmed across repeated runs because schedule tuning can have variance

## Metric Alignment

This submission should be judged only by the official challenge metric and constraints:

- `val_bpb` on the official FineWeb validation split
- official artifact byte accounting
- official evaluation legality
- official wallclock budget

Internal proxy metrics used during development are not submission evidence.

## Exact Setup

Repository base:

- `openai/parameter-golf`

Dataset path:

- `./data/datasets/fineweb10B_sp1024`

Tokenizer path:

- `./data/tokenizers/fineweb_1024_bpe.model`

Hardware used for final confirming run(s):

- `1x NVIDIA RTX 3090 (Runpod)`

## Exact Command

```bash
# Confirming run #1 (best metric)
RUN_ID=lr_warmdown_v1_run2 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
WARMDOWN_ITERS=900 \
MAX_WALLCLOCK_SECONDS=580 \
python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py | tee lr_warmdown_v1_run2.log

# Confirming run #2
RUN_ID=lr_warmdown_v1_run3 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
WARMDOWN_ITERS=900 \
MAX_WALLCLOCK_SECONDS=580 \
python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py | tee lr_warmdown_v1_run3.log
```

Confirming runs used identical settings, differing only in `RUN_ID`.

## Results

Public baseline reference:

- baseline `val_bpb`: `1.2244`
- source: official challenge README

Official baseline schedule defaults:

- `ITERATIONS=20000`
- `WARMUP_STEPS=20`
- `WARMDOWN_ITERS=1200`

Reproduced baseline reference:

- reproduced baseline `val_bpb`: `1.59422270`
- reproduced baseline artifact bytes (int8+zlib): `9191991`

Submission result:

- best `val_bpb`: `1.56446831` (run2)
- mean `val_bpb` across confirming runs: `1.56591334` (run2+run3)
- standard deviation across confirming runs: `0.00144503` (population stdev)
- compressed model bytes: `9542029`
- counted code bytes: `47686`
- total artifact bytes: `9589715`
- wallclock seconds: `581.264`

## Statistical Evidence

List confirming runs:

- `run2.log`: `val_bpb=1.56446831`, `wallclock_seconds=581.264`, `artifact_bytes=9589715`
- `run3.log`: `val_bpb=1.56735837`, `wallclock_seconds=580.980`, `artifact_bytes=9587743`

If this becomes a new SOTA claim, include the exact improvement margin and significance calculation.

## Files Included

- `submission.json`
- `train_gpt.py`
- `run2.log`
- `run3.log`
- `requirements.txt` if needed

## Reproducibility Notes

This submission is intended to keep the official baseline behavior unchanged except for schedule tuning.

In practice, for this repository that means:

- change `WARMUP_STEPS`
- change `WARMDOWN_ITERS`
- keep baseline optimizer LR values fixed

Confirm before submitting:

- same dataset/tokenizer path across baseline and tuned runs
- same evaluation method as official baseline unless explicitly justified
- same artifact accounting method
- frozen command recorded exactly
- baseline optimizer LR values unchanged

## Risks / Caveats

- schedule-only gains may be small
- variance can make a weak win look stronger than it is
- this run should not be claimed as meaningful unless it beats the reproduced baseline on official `val_bpb`
