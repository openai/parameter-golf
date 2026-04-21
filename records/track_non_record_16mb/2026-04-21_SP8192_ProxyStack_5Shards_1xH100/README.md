# SP8192 proxy-stack (5 shards, 1×H100 screening) — Non-record submission

This is a small, reproducible **screening** submission showing that simply switching the Track-P proxy stack from **SP4096 → SP8192** improves validation BPB under a fixed 10-minute wallclock budget on **1×H100**, trained on **5 FineWeb training shards**.

- **Track**: non-record (screening / grant experiments)
- **Base trainer**: `records/track_10min_16mb/2026-04-01_Vocab4096_MLPMult4_WD085/train_gpt.py` (env-driven)
- **Change vs base**: `VOCAB_SIZE=8192` (and matching cached dataset variant)
- **Budget**: `MAX_WALLCLOCK_SECONDS=600` (with GPTQ reserving ~10s)
- **Train shards**: 5

## Results (3-seed)

Metric notes:
- **Pre-quantization post-EMA** isolates raw model quality before GPTQ export.
- **Sliding BPB** below uses `final_int6_sliding_window` from the trainer logs.

| Seed | Steps @ cap | Pre-quant post-EMA `val_bpb` | `final_int6_sliding_window val_bpb` | Total submission size (int6+brotli) |
|------|-------------|------------------------------:|------------------------------------:|------------------------------------:|
| 0    | 1033        | 1.247118 | 1.247542 | 14,108,633 |
| 42   | 1031        | 1.245035 | 1.245657 | 14,101,451 |
| 1337 | 1035        | 1.245422 | 1.246063 | 14,124,187 |
| **Mean** | | **1.245858** | **1.246421** | **14,111,424** |

## How to run

You must have the cached dataset + tokenizer for SP8192 available under `data/` (see repo root `README.md`), and FlashAttention 3 installed (same requirement as the base trainer).

Example:

```bash
cd records/track_non_record_16mb/2026-04-21_SP8192_ProxyStack_5Shards_1xH100
SEED=1337 RUN_ID=screen_sp8192_1337 MAX_WALLCLOCK_SECONDS=600 VOCAB_SIZE=8192 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Files

- `train_gpt.py`: thin launcher that executes the base trainer with env vars.
- `train_seed*.log`: end-of-run excerpts for the three seeds used above.
- `submission.json`: metadata for this non-record screening submission.

