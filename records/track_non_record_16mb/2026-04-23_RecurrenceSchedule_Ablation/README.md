# Recurrence Schedule Ablation (Non-Record Research)

## Scope
This PR is a research/ablation contribution, not an official leaderboard submission.

Base stack:
- April 5 SP8192 stack (`2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2`)
- Same topology, optimizer, quantization/export path, eval settings
- Only recurrence schedule changed

## Hypothesis
A smooth recurrence curriculum is useful, but early/wide ramps overpay compute and can move the optimization shock earlier.
A later/narrower ramp should preserve most of the stability benefit while recovering step count.

## Controlled Variables
Kept fixed across runs:
- tokenizer, model architecture, looped layers (4–5)
- Muon/Adam settings
- QK/rope settings
- quantization and compression path
- no TTT, no parallel residual changes

## Schedules
- V1: hard switch control at 0.50
- V2: ramp (0.44 -> 0.50 -> 0.56)
- V3: ramp (0.485 -> 0.500 -> 0.515)

## Results (seed=1337, 8xH100, 600s cap)
| Run | Schedule | Steps | prequant_post_ema_val_bpb | quantized_val_bpb | quantized_sliding_bpb |
|---|---|---:|---:|---:|---:|
| V1 | hard @0.50 | 4956 | 1.09116446 | [fill] | [fill] |
| V2 | 0.44/0.50/0.56 | 4716 | 1.09186485 | [fill] | [fill] |
| V3 | 0.485/0.500/0.515 | 4773 | 1.09183070 | 1.10314309 | 1.08648521 |

Transition diagnostics:
- V2 ramp_start delta: 0.103111
- V3 ramp_start delta: 0.090919
- V2 full_on delta: 0.001988
- V3 full_on delta: 0.009417

Throughput diagnostics (mean step_ms):
- V2 no/ramp/full: 104.659 / 138.204 / 137.372
- V3 no/ramp/full: 105.257 / 136.589 / 137.992

## Takeaway
- V3 improved over V2 (more steps + slightly better prequant bpb).
- Early wide ramp is clearly worse for this stack.
- Late narrow ramp is a better recurrence schedule direction.

## Compute Grant Request
I request additional compute credits to run:
- 3-seed confirmation for V1/V3
- 1–2 additional narrow-ramp variants around 0.50
- same fixed stack and reporting format
