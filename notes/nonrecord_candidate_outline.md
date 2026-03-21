# Non-Record Candidate Outline

## Candidate name

- `TPI-001 Sliding Eval First`

## One-line thesis

- Improve local context use at evaluation before changing model size, tokenizer, or optimizer behavior.

## Why non-record first

- This branch is meant to validate the thesis with a minimal, explainable eval-policy change.
- It does not yet have enough empirical evidence for a stronger submission claim.

## What changed

- Added an optional sliding-window validation policy controlled by `EVAL_STRIDE`.
- Left baseline behavior intact when the policy knob is not engaged.

## What evidence exists

- Theory packet and implementation plan are aligned.
- `train_gpt.py` compiles successfully.
- Code-size delta is measured provisionally.

## What remains before records submission

- Run a real baseline/candidate comparison.
- Measure eval-time overhead and total artifact size from real artifacts.
- Convert the current notes into a polished README plus submission metadata if the result is worth keeping.
