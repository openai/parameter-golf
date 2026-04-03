# BW14_BigSwing_2k — Hypothesis

Parent baseline: BW13 control (`tap-off Nightcrawler`) at 2k gate:

- `int6_sw_bpb=1.27151667`
- `step_ms=156.63`

Primary objective: find another **phase-shift** improvement (target class: up to ~0.01),
not a micro delta.

## Why this leg

Recent legs showed:

- Quant policy changes are real but small (`~0.0017–0.0020`).
- Anchor-on-tap-off is negative.
- The largest observed jumps in this lineage came from architecture phase shifts
  (e.g., 4F->5F).

Archived choke notes also indicate a potentially large but unstable regime in choke routing,
particularly full-width per-loop routing (`choke_dim=512`) and bypassed variants.

## Test arms (WINDOW-only, retrain required)

| Arm | Change vs control | Intent |
|-----|-------------------|--------|
| `BW14BS-00` | control (tap-off Nightcrawler) | repin baseline |
| `BW14BS-01` | `NUM_FLAT_LAYERS=6` | depth phase shift beyond 5F |
| `BW14BS-02` | `CRAWLER_MLP_CHOKE_DIM=128`, `CHOKE_SHAPE=flat` | narrow per-loop routing with low cold-param cost |
| `BW14BS-03` | `CRAWLER_MLP_CHOKE_DIM=512`, `CHOKE_SHAPE=flat` | full-width per-loop routing (highest upside/risk) |
| `BW14BS-04` | `CRAWLER_MLP_CHOKE_DIM=128`, `CHOKE_SHAPE=residual` | shared bypass + per-loop delta to reduce cold-start burden |

## Promotion policy (to full 600s)

- Big-swing promote: `delta_vs_control <= -0.0060`
- Secondary promote: `-0.0060 < delta_vs_control <= -0.0030`
- Guardrails: artifact <= 16MB and no catastrophic speed regression.

## Run

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-01_BW14_BigSwing_2k/run_ablation_sequence.sh
```
