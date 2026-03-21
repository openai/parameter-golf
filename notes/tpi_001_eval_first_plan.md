# TPI-001 Eval-First Plan

## Chosen implementation

- Add a minimal sliding-window validation policy gated by `EVAL_STRIDE`.
- Preserve the baseline evaluation when `EVAL_STRIDE` is unset or greater than/equal to `TRAIN_SEQ_LEN`.

## Target files

- `train_gpt.py`
- `notes/tpi_001_eval_first_plan.md`

## Expected benefit

- Score tokens with richer recent context at evaluation time.
- Test the thesis through a local eval-policy change rather than through extra static knowledge or larger model state.

## Expected risk

- Evaluation takes longer because windows overlap.
- The gain may be evaluation-only and not transfer to a stronger submission story unless the README stays explicit about that.

## Budget risk

- Model bytes should stay unchanged.
- Code bytes should increase only slightly because the delta is confined to the eval path and one logits-return path.

## Why this is minimal

- One policy knob.
- One main code file.
- No tokenizer change.
- No optimizer or train-loop redesign.
- Easy to explain as "same model, different local context usage at eval."
