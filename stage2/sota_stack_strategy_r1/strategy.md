# Stage 2 SOTA Stack Strategy R1

This is the first-principles direct strategy.

## Classification Of `hypotheses_v2`

The good part of [hypotheses_v2.md]( nanoevolve/pgolf/parameter-golf/stage2/h100_matrix_r1/hypotheses_v2.md) is the pivot:

- stop treating the seq4096 trunk as the frontier
- move the control to the record-like SOTA stack

But the hypotheses in that note are not all equally valid for the same screen horizon.

### Good For Early Training Screens

- smaller batch / more steps
- higher Muon momentum
- longer warmdown
- Muon weight decay
- Muon backend steps
- the combined interaction test

### Not Good For The Same Early Training Screen

- denser sliding eval

Reason:

- it is an eval-policy hypothesis, not a train-dynamics hypothesis
- it contaminates the short screen with eval-time cost
- it should be screened later on a shared checkpoint, not in the first training pack

## The Active Sequence

Use one direct script and one direct sequence:

1. `sanity`
   - `8 x 1xH100`
   - same pack as the screen
   - purpose: crash / legality / pathological throughput

2. `screen`
   - `8 x 1xH100`
   - `2` controls + `6` candidates
   - purpose: prune by train-observable signal

3. `final_single`
   - top `1` by default
   - `1xH100` for the full `600s`
   - purpose: verify the survivor on a full single-GPU run

4. `champion_8x`
   - optional
   - single best survivor only
   - purpose: real challenge-aligned confirmation

## The Screen Pack

| Slot | Type | Mechanism |
| --- | --- | --- |
| `R0A` | control | record-like stack |
| `R0B` | control repeat | noise estimate |
| `H1` | candidate | geometry / step budget |
| `H2` | candidate | optimizer dynamics |
| `H3` | candidate | schedule / quant robustness |
| `H4` | candidate | regularization / quant robustness |
| `H5` | candidate | optimizer compute / per-step quality |
| `H6` | candidate | interaction / portability test |

## Why This Is Better

- every candidate is measured against the right baseline
- the pack is anchored on the actual frontier regime
- the early screen only contains hypotheses that can plausibly show signal in training
- control repeat is built in, so small deltas are interpretable
- the sequence is explicit enough to run directly without reconstructing the logic by hand

## Deferred Lane

Eval-only and export-only hypotheses still matter, but they should be screened later on shared checkpoints.

That includes:

- denser sliding eval
- export precision variants
- quantizer clip / scale policy

Those are not part of the first training screen because the first job is to identify the best training-side branch on the SOTA stack.
