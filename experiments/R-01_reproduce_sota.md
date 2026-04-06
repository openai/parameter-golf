# R-01: Reproduce SOTA (1.1147 BPB)

## Metadata
- **Date**: 2026-04-06
- **Branch**: exp/reproduce-sota
- **Parent**: main (uses leader's code verbatim)
- **Priority**: P0 (MUST DO FIRST)
- **Estimated runs**: 3 seed runs on 8xH100
- **Estimated cost**: ~$10 (3 runs x 10min x $20/hr)

## Goal
Run the leader's exact code (`records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`)
on our 8xH100 infrastructure. Confirm we can match their reported numbers.

## Expected Results (from their logs)
| Seed | Expected BPB | Expected Steps | Expected ms/step | Expected Artifact |
|------|-------------|----------------|------------------|-------------------|
| 314  | 1.1151      | 6,927          | 86.6             | 15,863,278        |
| 42   | 1.1144      | 6,922          | 86.7             | 15,984,850        |
| 999  | 1.1148      | 6,917          | 86.8             | 15,876,310        |
| Mean | 1.1147      |                |                  |                   |

## Reproduction Tolerance
- BPB within +/- 0.002 of their reported number per seed
- If ms/step differs significantly (>5ms), indicates hardware/driver differences
- If BPB differs by >0.003, investigate (wrong deps, different FA3 version, etc.)

## Prerequisites
- [ ] 8xH100 SXM pod on RunPod
- [ ] PyTorch 2.9.1+cu128 installed
- [ ] Flash Attention 3 (Hopper) installed: `pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291`
- [ ] sentencepiece, zstandard installed
- [ ] Training data downloaded: `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80`
- [ ] Verify deps: `python3 -c "from flash_attn_interface import flash_attn_func; import sentencepiece, zstandard; print('deps OK')"`

## Run Commands
```bash
cd /path/to/parameter_golf

# Copy SOTA script to working directory
cp records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py ./train_gpt_sota.py

# Seed 314
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt_sota.py

# Seed 42
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt_sota.py

# Seed 999
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=999 \
torchrun --standalone --nproc_per_node=8 train_gpt_sota.py
```

## What to Watch in Logs
Key log lines to match against their logs:
- `model_params:27067484` (exact match expected)
- `XSA:last_11 active_layers:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- `world_size:8 grad_accum_steps:1`
- `train_batch_tokens:786432 train_seq_len:2048`
- `step:4000/20000 val_loss:2.0348 val_bpb:1.2051` (mid-training checkpoint)
- `swa:start step:~6150`
- `late_qat:enabled step:~6335`
- `gptq:generated 64 sequences in ~196s`
- Final `final_int6_sliding_window_exact val_bpb:1.11508120` (seed=314)

## Results

### Seed 314
- Date:
- val_bpb (sliding):
- Artifact size:
- Steps / ms per step:
- Match: YES/NO (within tolerance?)
- Notes:

### Seed 42
- Date:
- val_bpb (sliding):
- Artifact size:
- Steps / ms per step:
- Match: YES/NO
- Notes:

### Seed 999
- Date:
- val_bpb (sliding):
- Artifact size:
- Steps / ms per step:
- Match: YES/NO
- Notes:

### Summary
- 3-seed mean:
- Std:
- Delta vs leader's 1.1147:
- Reproduction: SUCCESS / PARTIAL / FAILED

## Post-Mortem
### Discrepancies found:
### Hardware differences:
### Action items before Phase 1:
