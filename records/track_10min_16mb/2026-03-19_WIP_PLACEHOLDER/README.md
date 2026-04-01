# WIP Placeholder Submission

This is a working submission scaffold for the 10-minute / 16MB track.
Rename this folder later once you have final results.

## Goal

- Track: `track_10min_16mb`
- Objective: improve `val_bpb` while staying under the 16,000,000-byte artifact cap
- Budget: reproducible <= 10 minutes training on 8xH100 (SXM)

## Current Status

- Status: work in progress
- Baseline script source: root `train_gpt.py` copied into this folder
- Final metrics: pending

## Planned Changes

- [ ] Model/optimizer changes
- [ ] Data/tokenizer changes (if any)
- [ ] Eval method changes (if any)
- [ ] Compression/export changes (if any)

## Run Command (Template)

```bash
RUN_ID=wip_placeholder \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-19_WIP_PLACEHOLDER/train_gpt.py | tee records/track_10min_16mb/2026-03-19_WIP_PLACEHOLDER/train.log
```

## Required Files Checklist

- [x] `train_gpt.py`
- [ ] `train.log` (generate after run)
- [x] `README.md`
- [x] `submission.json`
- [ ] extra seed logs for SOTA significance (if needed)

## Results (Fill In)

Primary run:
- seed: 
- steps reached in 600s: 
- pre-quant `val_bpb`: 
- post-quant `val_bpb` (`final_int8_zlib_roundtrip_exact`): 
- `Total submission size int8+zlib`: 

Extra reproducibility runs (if claiming SOTA):
- `train_seedXXXX.log`: 
- `train_seedYYYY.log`: 

## Notes

- Keep logs and code in this folder so the PR is self-contained.
- If tokenizer or dataset changes are made, include proof that `val_bpb` is computed correctly.
