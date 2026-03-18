# 8. Make the Model Modestly Larger and Train It on Fewer Tokens

## Category

Architecture changes

## Why

This baseline is tiny for `8xH100`, and the README's 4-hour run beating it with essentially the same family suggests the 10-minute script is under-optimized more than fundamentally well-sized.

The proposed sweep is around slightly larger models rather than assuming `9x512` is compute-optimal. Concrete branches mentioned in `gpt-pro.md` are:

- `12x512`
- `14x448` or `14x480`
- `12x576`

This pairs naturally with KV-head reduction and an untied head.

## Tradeoffs

- Speed: slower per step, offset by fewer steps or shorter train context
- Size: moderately larger
- Complexity/risk: low-moderate

## Repo Fit

This is a clean sweep in the current codebase.
