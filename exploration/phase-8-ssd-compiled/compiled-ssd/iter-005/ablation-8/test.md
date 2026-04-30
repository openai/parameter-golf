# Ablation 8: Remove autocast(enabled=False) from _prepare_ssd_inputs

## Hypothesis

iter-005.5 gets val_bpb=1.98 despite 2x throughput vs iter-003.5 (val_bpb=1.600).
The regression may be caused by `torch.amp.autocast(device_type=..., enabled=False)`
wrapping ALL of `_prepare_ssd_inputs` (line 866 of iter-005.5/train_gpt.py).

This forces every tensor reshape, validation, and downstream computation that inherits
these tensors to run in fp32. Since `_prepare_ssd_inputs` is called by both
`_ssd_chunk_pytorch` and `_ssd_chunk_triton`, this effectively disables bf16 for the
entire SSD computation path.

iter-003.5's `_ssd_chunk` had `@torch.compiler.disable` but NO autocast override.
It ran in bf16 under the caller's autocast context, with only `_segsum` doing
`.float()` internally for cumsum numerical stability.

## Change

- Remove `with torch.amp.autocast(device_type=X.device.type, enabled=False)` from
  `_prepare_ssd_inputs`. Keep the method body at the same indentation (no wrapper).
- Keep `@torch.compiler.disable` on all SSD methods.
- The cumsum in `_ssd_chunk_pytorch` already uses `.float()` explicitly
  (`A_cumsum = torch.cumsum(A.float(), dim=-1)`), so numerical precision is preserved
  where it matters.

## Expected outcome

- SSD computation runs in bf16 (matching iter-003.5 behavior)
- ~2x speedup in SSD forward/backward (bf16 tensor cores vs fp32)
- val_bpb should improve toward 1.60 if autocast was the regression cause
- If val_bpb does NOT improve, the regression is architectural (vertical carry,
  state layout [B,H,N,P] vs [B,H,P,N], etc.)

## What to watch for

- NaN/Inf in loss (would indicate bf16 overflow in SSD math)
- If stable: compare tokens/sec to iter-005.5 baseline
- If unstable: the autocast was there for a reason, and we need selective fp32
  only for the cumsum/exp path (which is already handled by `.float()`)
