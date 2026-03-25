# H7: Noisy QAT + Skiptrace — Fix the Quant Gap

## Question
Does Noisy QAT (from PR #363) fix the crawler bank's quantization penalty,
making skiptrace viable on the competition frame?

## Prediction
The crawler bank's quant gap (0.059-0.066) is a major reason it loses to
the control. PR #363 showed Noisy QAT collapses quant gap from 0.37 to
0.002 on looped architectures by injecting calibrated uniform noise during
training. If applied to the crawler bank block only, it should:
- Reduce quant gap to <0.01
- Combined with skiptrace's ~1.5% overhead, make the bank net positive

## Implementation
Add to the crawler bank block's forward pass during training:
```python
with torch.no_grad():
    amax = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    step_size = amax / 127.0  # int8 scale
noise = (torch.rand_like(w) - 0.5) * step_size
w = w + noise
```
~10 lines of code in the Block class, gated by a flag.

## Arms (8L/384d, 0.25 scale)
| Arm | Config |
|-----|--------|
| Control | No bank |
| Skiptrace | Bank cad=10, no Noisy QAT |
| Skiptrace + NoisyQAT | Bank cad=10, Noisy QAT on bank block |

## Prerequisites
- H5 results (need to know if skiptrace shows any signal first)
- Code change: add noisy forward to crawler bank Block

## Status
BLOCKED on H5. Needs code change.

## Verdict
_To be filled after runs._
