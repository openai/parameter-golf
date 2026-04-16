# Record: SP8192 + 3-Layer Recurrence + Hard Onset — val_bpb 1.08625

**val bpb: 1.08625** (3-seed mean, std=0.0023)

| Seed | Steps | Pre-quant BPB | Post-quant BPB | **Sliding BPB** | Artifact |
|-|-|-|-|-|-|
| 314  | 5258  | 1.08438 | 1.10124 | **1.08424** | 15,998,501 |
| 1337 | 5247  | 1.08536 | 1.10271 | **1.08582** | 15,857,563 |
| 42   | 5033  | 1.08683 | 1.10559 | **1.08868** | 15,974,578 |
| **Mean** | | 1.08553 | 1.10318 | **1.08625** | 15,943,547 |

## Changes

This script builds on [#1394](https://github.com/openai/parameter-golf/pull/1394). The main changes are:

* **Hard onset for depth recurrence.** Recurrence on layers 3–5 activates as a binary switch at step 3000 (`RECUR_HOMOTOPY=0, RECUR_START_STEP=3000`), replacing the gradual sigmoid ramp in #1394. The model trains at full non-recurrent throughput for the first 3000 steps, then runs the shared-parameter recurrence sequence for the remaining ~2200 steps.
* **Removed mid-training validation.** `VAL_LOSS_EVERY=99999` eliminates validation passes during training, recovering approximately 200 additional training steps within the 600s wall-clock budget.
