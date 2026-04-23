# Idea 034b — Compress `031A` into a `025c`-style frozen per-pass carry

## Thesis

`034` tests the most faithful freeze of `031A`: keep the learned direct-carry
object in its native form and freeze it as buffers.

But we also want the simpler deployment-style question:

- can the useful part of `031A` be compressed back into a smaller `025c`-style
  per-pass `alpha/beta` object?

This is the more apples-to-apples comparison against the older frozen carry arc.

## Why this is separate from `034`

`034` asks:

- does native frozen direct-carry work?

`034b` asks:

- can a simpler per-pass `alpha/beta` approximation of `031A` work nearly as
  well?

So `034` is the faithful freeze.
`034b` is the compressed freeze.

## Compression target

Map the `031A` late snapshot into a per-pass `025c`-style object:

- `beta[pass, dst]`
- `alpha[pass, dst, src]`

with total parameter count:

- `beta`: `2 x 3 = 6`
- `alpha`: `2 x 3 x 3 = 18`
- total = `24`

instead of `031A`'s `33` frozen scalars.

## Why it is worth testing

If `034b` is close to `034`, that is attractive:

- simpler mechanism
- easier comparison to `025c`
- smaller and cleaner frozen carry object

If `034b` loses clearly to `034`, then the extra structure in native direct-carry
was load-bearing.
