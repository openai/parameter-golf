# No-looping ablation on the SP8192 SOTA stack (5 shards, 1×H100 screening) — Non-record submission

This folder captures the **“no looping”** ablation used during grant screening: run the current SP8192 SOTA training stack with **layer looping disabled**.

- **Track**: non-record (screening / grant experiments)
- **Base trainer**: `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py`
- **Ablation**: set `NUM_LOOPS=0` (disables depth recurrence / looping)
- **Budget**: `MAX_WALLCLOCK_SECONDS=600`
- **Train shards**: 5

## Results (3-seed)

Metric notes:
- **Pre-quantization post-EMA** isolates model quality before export.
- **`quantized_sliding_window`** is the post-quant sliding-window BPB reported by this trainer.

| Seed | Steps @ cap | Pre-quant post-EMA `val_bpb` | `quantized_sliding_window val_bpb` | Total submission size (quantized+brotli) |
|------|-------------|------------------------------:|-----------------------------------:|-----------------------------------------:|
| 0    | 658         | 1.327667 | 1.317445 | 16,033,831 |
| 42   | 724         | 1.291048 | 1.280317 | 16,034,416 |
| 1337 | 724         | 1.289564 | 1.278652 | 16,034,548 |
| **Mean** | | **1.302760** | **1.292138** | **16,034,265** |

## How to run

```bash
cd records/track_non_record_16mb/2026-04-21_NoLooping_SOTAStack_5Shards_1xH100
SEED=1337 RUN_ID=no_looping_1337 MAX_WALLCLOCK_SECONDS=600 NUM_LOOPS=0 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Notes

- This submission uses a **thin launcher** that sets the no-looping env var and then executes the base record trainer.
- Training/eval dependencies should match the base record (FlashAttention 3, etc.).

