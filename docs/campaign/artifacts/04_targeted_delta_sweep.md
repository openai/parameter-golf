# Session 04 Closeout: Targeted Delta Sweep

Date: 2026-03-29

## Goal

Test a few small isolated deltas on top of the measured Session 03 anchor and determine whether any should graduate into the mainline stack.

Anchor reference:

- Sliding s64 val_bpb: `1.12904446`
- Pre-quant EMA val_bpb: `1.14472403`
- Int6 roundtrip val_bpb: `1.15247273`
- Steps: `6564`
- Step average: `91.37 ms`
- Artifact: `15751324` bytes

## Delta 1: GPTQ-lite percentile clip search

Result:

- Sliding s64 val_bpb: `1.12941356` (`+0.00036910`, worse)
- Pre-quant EMA val_bpb: `1.14520403`
- Int6 roundtrip val_bpb: `1.15277272` (`+0.00029999`, worse)
- Steps: `6565`
- Step average: `91.37 ms`
- Artifact: `16219752` bytes — over the `16000000` byte cap

Conclusion:

- Failed.
- GPTQ-lite clip search hurt zstd compressibility more than it helped quantization quality.
- It is not a viable next step on this anchor.

## Delta 2: LeakyReLU^2

Result:

- Sliding s64 val_bpb: `1.12904123` (`-0.00000323`, effectively identical)
- Pre-quant EMA val_bpb: `1.14438546` (slightly better)
- Int6 roundtrip val_bpb: `1.15222198` (slightly better)
- Steps: `6511` (`-53`)
- Step average: `92.09 ms` (`+0.72 ms`)
- Artifact: `15582968` bytes (`-168356` bytes)

Conclusion:

- Neutral / tie.
- LeakyReLU^2 marginally improves quantization-friendliness and artifact size, but the slower step time cancels the small gain in the headline metric.
- Keepable as a future stack component, but not a standalone graduating delta.

## Session 04 Decision

Session 04 is complete.

Outcome:

- `1` failed delta
- `1` neutral delta
- no standalone graduating delta

Interpretation:

- The narrow isolated-delta sweep was still useful.
- It ruled out one export-side hypothesis and identified one stackable-but-not-standalone model tweak.
- The next highest-value work is no longer another tiny delta by default.

## Session 05 Handoff

Open Session 05 with three linked tracks:

1. Throughput audit
   - explain the gap from anchor `91.37 ms` to the local public `83.4 ms` stack
   - investigate FA3 portability first
2. Pre-TTT base enhancement audit
   - compare the anchor to the local `1.1194` record
   - rank the easy portable stack features
3. TTT correctness audit
   - audit legality, cost, and portability of score-first TTT

The local public `1.1194` stack should be treated as the primary comparison target for Session 05.
