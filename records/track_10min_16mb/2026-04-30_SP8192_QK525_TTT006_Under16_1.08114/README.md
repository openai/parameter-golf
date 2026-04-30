# SP8192 QK5.25 Legal TTT LR0.006

3-seed mean val_bpb: 1.08113832. All artifacts are under the 16,000,000 byte cap.

## Results

| seed | val_bpb | val_loss | artifact bytes |
|---|---:|---:|---:|
| 42 | 1.08053255 | 2.79112779 | 15994309 |
| 314 | 1.08130302 | 2.79311801 | 15994445 |
| 999 | 1.08157940 | 2.79383193 | 15991302 |
| mean | 1.08113832 | | |

## Reproduction

Run from repo root:

SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.006 TTT_EPOCHS=3 torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-04-30_SP8192_QK525_TTT006_Under16_1.08114/train_gpt.py

Use SEED=314 and SEED=999 for the other logs.
