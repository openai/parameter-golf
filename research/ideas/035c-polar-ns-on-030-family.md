# Idea 035c — Polar NS on the original `030` alpha/beta family

> Obsolete slot. The active continuation is now `035d` then `035e`.

## Thesis

The second-ranked `#1779` follow-up is the optimizer-side refinement:

- keep the stronger original `030` alpha/beta family fixed
- replace stock Muon's repeated fixed Newton-Schulz tuple with the 5 Polar
  Express per-iteration tuples from PR `#1344`
- first test it as a `4×H` pre-quant screen with no TTT

## Benchmark

Primary `4×H` alpha/beta-family reference:

- `026` screen seed `314`: pre-quant `1.06770372`

Direct schedule/optimizer siblings:

- `035` = `MIN_LR=0.10`
- `035b` = loop-onset plateau

## First question

Can Polar NS alone beat `1.06770372` on the original `030` family in `4×H`
screen form?
