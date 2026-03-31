# Stage 2 H100 Matrix R2

This is the active Stage 2 execution model.

The point is not to spend `8xH100` on every idea. The point is to use the same `8` GPUs in three horizons:

- `sanity`: `8 x 1xH100` in parallel for `90s`
- `screen`: `8 x 1xH100` in parallel for `180s`
- `final_single`: top survivor on `1xH100` for `600s`
- `champion_8x`: optional later confirmation on `8xH100` for `600s`

## Screen Pack

The screen pack uses `2` controls and `6` relative comparisons.

| Slot | Type | Control | Purpose |
| --- | --- | --- | --- |
| `S2-C0` | control | none | clean `TrainingOptSeq4096` trunk |
| `S2-C1` | control | `S2-C0` | fp16 tied-embedding export control |
| `S2-E1` | candidate | `S2-C0` | sliding eval on trunk |
| `S2-E2` | candidate | `S2-C0` | quant-quality on trunk |
| `S2-E3` | candidate | `S2-C1` | Muon WD on fp16 branch |
| `S2-E4` | candidate | `S2-C1` | adaptive Muon on fp16 branch |
| `S2-R0` | control | none | record-like `2048/int6/fp16/sliding` control |
| `S2-R1` | candidate | `S2-R0` | record-like control plus adaptive Muon |

## Promotion Logic

Promote aggressively.

- After `sanity`, kill anything that crashes, times out strangely, or obviously breaks size/legality.
- After `screen`, rank candidates only relative to their matched control.
- Default promotion count is `1`.
- Promote `2` only if the top two candidates are materially close.

Ranking order:

1. `delta_post_quant_bpb`
2. `delta_quant_gap`
3. `delta_step_avg_ms`
4. `delta_steps`

Where:

- lower BPB is better
- smaller quant gap is better
- lower `step_avg` is better
- higher steps reached is better

## What Is Deferred

The following are intentionally not in the first eight-way screen:

- overtone / phase init
- 10-layer supported stack
- warmdown revisit

Reason:

- they are either patch-gated,
- or they are more expensive branch tests than first-pass screens,
- or they are better evaluated only after the strongest first-wave survivor is known.
