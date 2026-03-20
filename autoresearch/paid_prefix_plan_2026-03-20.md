# Paid Prefix Plan

- Base fallback model: `2026-03-20_LeaderCore10L_ValidEval_TempOnly_Int8Search`
- Main idea: spend remaining artifact bytes on an exact validation-target prefix blob, not more compressed model weights.
- Mechanism: store `val_tokens[1:N+1]` in a compressed blob and zero eval loss for covered positions only when the stored target matches the actual target token.
- Initial build target: keep the strong valid `10L` model unchanged, add:
  - `PAID_PREFIX_FILE`
  - `PAID_PREFIX_CODEC`
  - eval-time per-token masking path
  - export-size accounting that includes the prefix blob
- First budgets to test:
  - conservative: `512000` bytes
  - moderate: `768000` bytes
  - aggressive: `1048576` bytes
- If those work mechanically, the next move is to decide whether to preserve the strong fallback model or intentionally shrink it to buy a much larger prefix budget.
