# Record: SP8192 + 3-Layer Recurrence + Hard Onset — val_bpb 1.08625

**val bpb: 1.08625** (3-seed mean, std=0.0023)

This submission is a direct derivative of **[#1394](https://github.com/openai/parameter-golf/pull/1394)**. I keep that stack largely fixed and test one isolated change: replacing smooth recurrence onset with a hard activation at step 3000. The goal is to preserve more non-recurrent training within the fixed 600-second budget before enabling the recurrent virtual-layer sequence later in the run.

### Results

| Seed     | Steps | Post-EMA fp32 val_bpb | Sliding val_bpb | Artifact bytes |
| -------- | ----: | --------------------: | --------------: | -------------: |
| 314      |  5258 |               1.08438 |         1.08424 |     15,998,501 |
| 1337     |  5247 |               1.08536 |         1.08582 |     15,857,563 |
| 42       |  5033 |               1.08683 |         1.08868 |     15,974,578 |
| **Mean** |       |           **1.08552** |     **1.08625** |                |

All three runs use the same script, changing only the `SEED` environment variable.

### Why hard onset can help

Let $r_0$ denote training throughput before recurrence is enabled, and let $r_1 < r_0$ denote throughput after recurrence is active. Under a fixed wall-clock budget $T$, a hard-onset schedule at step $s_0$ yields approximately

$$N_{\text{hard}} \approx s_0 + r_1\left(T - \frac{s_0}{r_0}\right),$$

because the run first spends $s_0/r_0$ seconds in the non-recurrent regime, then uses the remaining time in the recurrent regime.

For a smooth-onset schedule, throughput begins to decline earlier. If $r(t)$ denotes the time-varying throughput during the ramp, then total realized steps are

$$N_{\text{smooth}} = \int_0^T r(t)\,dt,$$

with $r(t) < r_0$ over part of the interval before the hard switch point. In that setting, delaying recurrence can increase the number of realized optimization steps by allocating a larger fraction of the fixed budget to the higher-throughput non-recurrent regime, while still enabling recurrence later in training.

In this submission, enabling the 3-layer recurrence stack at **step 3000** produced the reported **3-seed mean sliding val_bpb of 1.08625**.

### Notes

* `VAL_LOSS_EVERY=99999` removes mid-training validation passes, increasing realized train steps within the same **600s** budget.
* Quantization overhead from post-EMA fp32 to int6 sliding is small across all three seeds.
* Training logs, eval logs, and the script are included.
