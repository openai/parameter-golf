# Experiment 009: Full 8xH100 Production Run — BLOCKED

## Status: FAILED — Thunder Compute PCIe topology too slow

## What Happened
Two attempts on Thunder Compute 8xH100 production:
1. **009a**: Hostname resolution blocked DDP init entirely. Fixed with `--rdzv-backend=static` + removing `device_id` from `init_process_group`.
2. **009b**: DDP worked but throughput was 231ms/step (5.4x slower than baseline's 43ms/step).

Root cause: Thunder Compute's 8xH100 uses **PCIe Host Bridge** between GPU pairs, not full NVLink mesh.
- Only pairwise NVLink: GPU0-1, GPU2-3, GPU4-5, GPU6-7
- Cross-pair comm goes over PCIe → massive DDP all-reduce overhead
- Result: ~2,600 steps in 10 min vs baseline's 13,780

## Partial Results (009b, clean baseline + softcap15)
| Step | val_bpb | train_time | step_avg |
|------|---------|------------|----------|
| 200  | 1.6615  | 46s        | 232ms    |
| 400  | 1.5161  | 93s        | 232ms    |
| 600  | 1.4479  | 138s       | 231ms    |

Killed at step 600. Would have reached ~step 2600 in 10 min.

## Cost
~$9.50 total for both 8xH100 attempts. Zero competitive results.

## Next Steps
Need Runpod or equivalent with proper NVLink mesh (DGX/HGX) for competition submission.
