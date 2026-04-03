## Lucky IV

Rascal II + brotli byte-shuffle + 24-step Context-Only SLOT test-time adaptation at inference.

## Results

| Seed | val_bpb (sliding window) | Steps | Size |
|------|--------------------------|-------|------|
| 444  | 1.09626897               | 6296  | 15,532,043 B |
| 4    | 1.09703170               | 6300  | 15,526,928 B |
| 300  | 1.09600210               | 6296  | 15,524,362 B |
| **mean** | **1.09643426**       |       | **15,532,043 B** |

Hardware: 8xH100 SXM · 600s wallclock · `bytes_code`: 123961

## Architecture changes

- SLOT_STEPS: 8 -> 24 (vs Slot Machine). Shared delta (1,1,dim), 24 AdamW optimization steps on context positions at eval time.

## Reproduce

```bash
SLOT_ENABLED=1 SLOT_STEPS=24 SKIP_GPTQ=1 SEED=444 python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-03_Lucky_IV_8xH100/train_gpt.py
```
