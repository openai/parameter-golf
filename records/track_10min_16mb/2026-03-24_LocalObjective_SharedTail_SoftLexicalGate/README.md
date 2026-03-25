# Local Objective + Shared Tail + Soft Lexical Gate

Submission scaffold for the current upgraded local-objective trainer.

## Recipe

1. Sliding-window validation / final evaluation
2. Shared recurrent tail with explicit diagnostics
3. Training-only mid auxiliary objective with ramp + decay
4. Stronger bigram defaults
5. Lexical shortcut controlled by soft `attn_gain_share` gating
6. Dead helper paths disabled (`xsa`, `late_route`, `relative_bias`)
7. Experimental packed-train batching removed from the submission copy

## Files

- `train_gpt.py`: exact trainer for the submission run
- `train.log`: copy the full run log here after the record run completes
- `submission.json`: fill in final metrics after the run completes

## Notes

- Source trainer copied from `train_gpt_upgrade_local_objective.py`
- Train-side packed batching was intentionally removed here.
  - The prior implementation was boundary masking on a fixed grid, not true document packing, so it was left out of the competition script and should only be revisited experimentally.
- Current copied trainer hash:
  - `7268edef691fc0da47dee4bf274023382bf2d66d07ec0038bd469d938769cac7`
- Recommended launch pattern:
  - `USE_CUDA_GRAPHS=0 torchrun --standalone --nproc_per_node=2 train_gpt.py`
