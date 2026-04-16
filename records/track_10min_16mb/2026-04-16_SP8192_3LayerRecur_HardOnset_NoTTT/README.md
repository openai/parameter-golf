## SP8192 + 3-layer recurrence + hard onset

**val_bpb 1.0862** (3-seed mean, std 0.0023) | **~15.94 MB** | 8x H100 SXM, 600s

The main gain comes from activating the 3-layer recurrence stack with a hard onset at step 3000, which outperformed smooth-onset variants in this line on the final artifact metric.

Built on the PR #1394 recurrence stack. The novel contribution is the onset schedule: hard activation of depth recurrence at a fixed training step rather than a gradual sigmoid ramp. This preserves more training steps within the wallclock budget because the model runs at full non-recurrent throughput until step 3000, then switches to the recurrent virtual-layer sequence for the remainder of training.

`VAL_LOSS_EVERY=99999` removes mid-training validation passes, increasing realized train steps within the same 600s wall-clock budget.

Recipe:
- SP8192
- 3-layer recurrence on layers 3-5
- hard onset at step 3000
- GPTQ int6 + brotli
- EMA 0.9965
- SDClip 12.85

## Results

| Seed | Steps | Post-EMA fp32 val_bpb | Sliding val_bpb | Artifact bytes |
|------|------:|----------------------:|----------------:|---------------:|
| 314  |  5258 |               1.08438 |       1.08424   |   15,998,501   |
| 1337 |  5247 |               1.08536 |       1.08582   |   15,857,563   |
| 42   |  5033 |               1.08683 |       1.08868   |   15,974,578   |
| Mean |       |               1.08552 |       1.08625   |                |

Artifact sizes 15.86-16.00 MB across seeds. Quantization overhead from post-EMA fp32 to int6 sliding is +0.0005 or less per seed.

## Deltas

Relative to our immediate recurrence baseline (D0 hard onset with standard val pauses, int6 sliding 1.09022): **-0.0040 BPB** mean improvement, primarily from the no-val-pause wallclock optimization.

Relative to PR #1394 (Clark, 1.0856): this result is within 0.0006 BPP of that line.

## Reproducibility

All three seeds run the same script with only the SEED env var changed. Training logs, eval logs, and the training script are included in this record directory. The run is fully deterministic given the same hardware class and seed.
