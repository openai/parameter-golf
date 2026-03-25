# H5: Cubric First Signal — Does Skiptrace Beat Every-Step Bank?

## Question
Can periodic crawler bank firing with learned decay injection match the
per-step quality of every-step firing at a fraction of the compute cost?

## Prediction
Skiptrace (cadence 10) will land between control and every-step bank on
per-step quality, but closer to control on step count. Net effect: skiptrace
beats every-step bank on sliding_window because the step count advantage
outweighs the small quality loss. If the learned decay parameter converges
to >0.5, the model is actively using the cached delta.

## Arms (8L/384d, 0.25 scale)
| Arm | Config | Expected overhead |
|-----|--------|-------------------|
| F (control) | No bank | 0% |
| G (every step) | Bank fires every step | ~15% |
| H (skiptrace) | Bank fires every 10, decaying injection | ~1.5% |

## Diagnostic Focus
- sliding_window BPB across all three arms
- Step count: H should be within 2% of F
- Monitor learned params: sigmoid(decay_logit) and sigmoid(inject_scale)

## Status
READY — scripts pushed.

## Verdict
_To be filled after runs._
