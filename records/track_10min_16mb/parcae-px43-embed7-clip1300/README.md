# Draft: parcae-px43-embed7-clip1300

**Status:** draft only. Do not submit this as a record PR without rerunning.

**3-seed mean val_bpb:** 1.08782605

| Seed | Sliding BPB | Train Time | Eval Time | Artifact Bytes |
|------|-------------|------------|-----------|----------------|
| 42 | 1.08802944 | 600.024s | 89.275s | 15,633,824 |
| 1337 | 1.08783878 | 600.117s | 89.174s | 15,630,505 |
| 2024 | 1.08760994 | 600.093s | 89.318s | 15,630,862 |
| Mean | 1.08782605 | 600.078s | 89.256s | 15,631,730 |

## Gate Status

This package is organized in the same shape as a Parameter Golf records-folder submission, but the phase 1 logs do not pass the record gate:

- artifact size is under 16,000,000 bytes
- all three required seeds completed and emitted `RUN_COMPLETE_DO_NOT_KILL submission_ready=1`
- final sliding eval is under 600 seconds
- training time is over 600 seconds in all three logs by 24-117 ms
- mean BPB does not beat the current leaderboard SOTA

## How to Run

```bash
pip install --break-system-packages \
  flash_attn_3 \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280

pip install --break-system-packages \
  'fused-softcap-ce @ git+https://github.com/anthony-maio/fused-softcap-ce.git@25e7ad6292cd1e837eef592f50e4d9f5990b6c84' \
  brotli zstandard sentencepiece numpy tqdm

DATA_DIR=./data \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Use the Mikeapedia SP8192 data layout:

```text
data/tokenizers/fineweb_8192_bpe.model
data/datasets/fineweb10B_sp8192/fineweb_train_*.bin
data/datasets/fineweb10B_sp8192/fineweb_val_*.bin
```

## Files

- `train_gpt.py`
- `submission.json`
- `PR_DESCRIPTION.md`
- `train_seed42.log`
- `train_seed1337.log`
- `train_seed2024.log`
- `requirements.txt`
- `GATE_REPORT.md`
