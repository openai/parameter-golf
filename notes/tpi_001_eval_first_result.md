# TPI-001 Eval-First Result

## What changed

- Added an `EVAL_STRIDE`-gated sliding-window validation mode to `train_gpt.py`.
- Preserved baseline validation when `EVAL_STRIDE` is unset or greater than/equal to `TRAIN_SEQ_LEN`.
- Kept the change inside the eval path and a small logits-return extension on `GPT.forward`.

## Delta size

- Functional code delta is concentrated in one file: `train_gpt.py`.
- Supporting docs and run notes were added under `notes/` and `runs/TPI-001/`.

## Execution result

- Full baseline vs candidate experiment: not run in this workspace
- Smoke validation: `python3 -m py_compile train_gpt.py` passed
- Runtime comparison result: unavailable

## Improvement status

- Score improvement: unverified
- Mechanism validity: syntactically valid and consistent with existing record precedent for sliding-window eval

## Risks

- Sliding eval can increase evaluation time materially.
- Without a real run, the actual score delta and wallclock impact are still unknown.
- The branch currently improves explainability more than evidence.

## README readiness

- Good
- The branch can be described cleanly as "same model, same training, better local context usage at evaluation."

## Next step

- Sharpen this branch with one real baseline/candidate run pair or a smaller remote smoke that reaches the eval path.
