## Slot Machine

Rascal II + brotli byte-shuffle compression + context-only SLOT 8-step test-time optimization.

## Results

| Seed | val_bpb (sliding window) | Steps | Size |
|------|--------------------------|-------|------|
| 444  | 1.10350531               | 6295  | 15,536,878 B |
| 300  | 1.10448947               | 6290  | 15,532,578 B |
| 42   | 1.10446734               | 6286  | 15,526,521 B |
| **mean** | **1.1042**           |       | **15,536,878 B** |

Hardware: 8xH100 SXM · 600s wallclock · `bytes_code`: 123960

## Architecture changes

- Compression: zstd-22 replaced with brotli-11 + byte-shuffle (stride=2), saving ~134KB model size
- Eval: added legal Context-Only SLOT with 8 optimization steps (~0.005 BPB improvement, no size increase)

## Reproduce

```bash
# From repo root, with flash-attention/hopper on PYTHONPATH
pip install brotli
SEED=444 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-03_Slot_Machine_8xH100/train_gpt.py
```
