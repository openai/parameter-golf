# BW12_Interaction_2k — Hypothesis

**Parent stack (verified):** Nightcrawler (`5F + TAP(shared)`, naive int6)

Primary goal: optimize interactions between crawler mechanisms and the current optimum stack,
while minimizing wasted retraining. All gate tests run **2000 steps**.

## Why this leg

Nightcrawler proves the current stack changed the frontier, but we still need clean interaction
isolation:

1. Which gains are from **5F depth** vs **TAP interaction**?
2. Does **anchor write-state** stack on Nightcrawler?
3. How much quantization lift comes from GPTQ policy when tested on **identical trained weights**?

## Test lanes (single orchestration script)

### Lane A — `WINDOW` (must retrain)

These alter training-time behavior/representations and require fresh 2k windows.

| Arm | Change vs control | Why retrain required |
|-----|-------------------|----------------------|
| `BW12INT-00` | control: Nightcrawler gate stack | baseline |
| `BW12INT-01` | `CRAWLER_TAP_DIM=0` | TAP changes forward path during training |
| `BW12INT-02` | `ANCHOR_DIM=32` | anchor path changes loop-state dynamics during training |

### Lane B — `POST_WINDOW` (sequential, no retrain)

These reuse **exact control weights** (`INIT_MODEL_PATH`) and only change post-training quant policy.

| Arm | Change vs control checkpoint | Why sequential after one window |
|-----|------------------------------|---------------------------------|
| `BW12INT-Q0` | naive int6 (`SKIP_GPTQ=1`) | reference quant pass on frozen weights |
| `BW12INT-Q1` | standard GPTQ (`SKIP_GPTQ=0, LOOP_AWARE_GPTQ=0`) | quant-only change |
| `BW12INT-Q2` | loop-aware GPTQ (`SKIP_GPTQ=0, LOOP_AWARE_GPTQ=1`) | quant-only change |

## Full-run-only promotion queue (600s, 8xH100)

Do not full-run everything. Promote only if gate signal clears noise floor.

- Promote any `WINDOW` arm with `delta_vs_control <= -0.0008`.
- Promote any `POST_WINDOW` quant arm with `delta_vs_control <= -0.0008`,
  then rerun that policy as a full training run (`SKIP_TRAIN=0`) for true leaderboard relevance.

## Run

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-01_BW12_Interaction_2k/run_ablation_sequence.sh
```

Optional speed lever for GPTQ calibration in gate context:

```bash
GPTQ_CAL_SAMPLES=96 GPTQ_CAL_SEQ_LEN=1024 \
SEED=444 NPROC_PER_NODE=4 \
bash crawler/2026-04-01_BW12_Interaction_2k/run_ablation_sequence.sh
```
