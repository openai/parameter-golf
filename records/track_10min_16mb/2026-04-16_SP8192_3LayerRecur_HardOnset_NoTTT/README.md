## SP8192 + 3-layer recurrence + hard onset

val_bpb 1.0862 (3-seed mean, std 0.0023) at ~15.94 MB on 8x H100 in 600s.

The main change here is recurrence activation: using a hard onset at step 3000
for the 3-layer recurrence stack gave the best final artifact metric in this line.

Recipe:
- SP8192
- 3-layer recurrence on layers 3-5
- hard onset at step 3000
- GPTQ int6 + brotli
- EMA 0.9965
- SDClip 12.85

Results:

| Seed | Steps | Post-EMA fp32 val_bpb | Sliding val_bpb | Artifact bytes |
|------|------:|----------------------:|----------------:|---------------:|
| 314  |  5258 |               1.08438 |       1.08424   |   15,998,501   |
| 1337 |  5247 |               1.08536 |       1.08582   |   15,857,563   |
| 42   |  5033 |               1.08683 |       1.08868   |   15,974,578   |
| Mean |       |               1.08552 |       1.08625   |                |

Notes:
- VAL_LOSS_EVERY=99999 removes mid-training validation passes and increases
  realized training steps within the same 600s wallclock budget.
- Hard onset outperformed the smooth-onset variants on the final artifact
  metric in this recurrence line.
