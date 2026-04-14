# Iterative Run Tracker — SKC-600 Competition

## Best Config So Far
- **Best val_bpb: 2.0830** (Run 2)
- **Best ternary_roundtrip_bpb: 2.1179** (Run 2)
- Budget: 4.95/16.00MB

## Run History

| Run | LR | Batch | Steps | val_bpb | roundtrip_bpb | Budget | Notes |
|-----|-----|-------|-------|---------|---------------|--------|-------|
| 1 | 0.022 | 16384 | 5151 | 2.1077 | - | - | Crashed at engram prune (device mismatch) |
| 2 | 0.015 | 16384 | 5161 | 2.0830 | 2.1179 | 4.95MB | First complete run. OOM in sliding eval (retry bug). Loss spikes at step 3001 (5.86) |
| 3 | 0.012 | 32768 | ? | ? | ? | ? | In progress. Lower LR + bigger batch for stability |

## Config Baseline (Run 2 — best so far)
- MODEL_DIM=600, NUM_LAYERS=12, NUM_HEADS=8, NUM_KV_HEADS=2
- MATRIX_LR=0.015, WARMDOWN_FRACTION=0.72
- BIGRAM_HASH_BUCKETS=8192, BIGRAM_HASH_DIM=64, ENGRAM_NUM_HEADS=2, ENGRAM_NUM_ORDERS=2
- TTT_LR=0.005, TTT_EPOCHS=3, TTT_SCOPE=skc_safe
- COMPILE_MODE=none, FP_STORAGE=fp4
- 2x A5000 24GB, ~115ms/step
