# BW13_TapOff_Anchor_GPTQ_2k — Hypothesis

**Parent evidence (BW12, seed=444):**

- `BW12INT-01` (tap off) improved over control: `delta=-0.001994`
- `BW12INT-02` (anchor32) improved over control: `delta=-0.001015`
- `BW12INT-Q1/Q2` improved over control: `delta=-0.002044` (Q1 and Q2 tied)

Primary goal: push interaction quality on the crawler optimum stack while keeping the 4x run efficient.

## Why this leg

BW12 proved all three levers can move score, but it did not isolate whether anchor gains stack on the tap-off baseline,
nor whether full GPTQ calibration budget is required for that improved stack.

## Test lanes (single orchestration script)

### Lane A — `WINDOW` (must retrain)

These alter training-time behavior and require fresh 2k windows.

| Arm | Change vs control | Why retrain required |
|-----|-------------------|----------------------|
| `BW13INT-00` | control: tap-off Nightcrawler (`CRAWLER_TAP_DIM=0`) | baseline |
| `BW13INT-01` | tap-off + `ANCHOR_DIM=32` | anchor changes loop-state training dynamics |
| `BW13INT-02` | tap-off + `ANCHOR_DIM=64` | checks whether larger anchor capacity helps or overfits at 2k |

### Lane B — `POST_WINDOW` (sequential, no retrain)

These reuse exact control weights (`SKIP_TRAIN=1`, `INIT_MODEL_PATH=<control final_model.pt>`).

| Arm | Change vs control checkpoint | Why sequential after one window |
|-----|------------------------------|---------------------------------|
| `BW13INT-Q0` | naive int6 (`SKIP_GPTQ=1`) | quant reference on frozen weights |
| `BW13INT-Q1` | standard GPTQ (`128x2048`) | quant-only policy test |
| `BW13INT-Q1L` | GPTQ-lite (`64x1024`) | checks if lower calibration budget preserves signal |

## Full-run-only promotion queue (600s, 8xH100)

Promote only if gate signal clears noise floor.

- Promote any `WINDOW` arm with `delta_vs_control <= -0.0008`.
- Promote any `POST_WINDOW` quant arm with `delta_vs_control <= -0.0008`,
  then rerun that policy with full training (`SKIP_TRAIN=0`) for leaderboard relevance.

## Run

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-01_BW13_TapOff_Anchor_GPTQ_2k/run_ablation_sequence.sh
```
