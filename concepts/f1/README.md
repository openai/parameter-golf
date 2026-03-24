# F1 Concept Baseline

This folder is a working copy of the race-car baseline from **PR #587** (`submission/xsa11-clean`), confirmed by you as the source to clone.

## Provenance

- PR: https://github.com/openai/parameter-golf/pull/587
- Head branch: `submission/xsa11-clean`
- Commit: `303192e9ac65fa1673de647b02d1bb7365c37198`
- Reported result (seed 1337): pre-TTT `1.1203`, TTT `1.1204`

## Files

- `train_gpt.py`: PR #587 base with F1 experimental knobs added
- `run.sh`: local runner wired to this folder's `train_gpt.py`
- `run_legal_lb.sh`: legal leaderboard profile (record-track tactics only)

## Run

```bash
SEED=1337 bash concepts/f1/run.sh
```

Legal leaderboard profile:

```bash
SEED=1337 bash concepts/f1/run_legal_lb.sh
```

## Teacher-Student + Extra Capacity Knobs

`train_gpt.py` now includes:

- `F1_CORR_RANK` / `F1_CORR_SCALE_INIT`: low-rank correction head (active at inference/export)
- `DISTILL_ENABLED` + `DISTILL_*`: post-train EMA teacher -> student distillation pass

Approx added params from correction head:

`extra_params ~= F1_CORR_RANK * (MODEL_DIM + VOCAB_SIZE)`

For `MODEL_DIM=512`, `VOCAB_SIZE=1024`:

- `RANK=224` -> ~344k params
- `RANK=256` -> ~393k params
- `RANK=288` -> ~442k params

## Legal Leaderboard Imports (Filtered)

Only strategies from official **record track** entries were imported:

- LeakyReLU-squared MLP (`MLP_ACT=leaky_relu_sq`, slope `0.5`)
- `XSA_LAST_N=4`
- `BIGRAM_VOCAB_SIZE=1536`
- legal score-first TTT profile (`TTT_FREEZE_BLOCKS=0`, `TTT_GRAD_CLIP=0.8`)

Excluded on purpose:

- non-record hardware strategies (for example A100/A800-only experiments)
- known illegal TTT recipes (including pre-eval TTT on validation labels)

## Suggested Profiles

Accuracy-first (use most of the spare model budget):

```bash
SEED=1337 \
F1_CORR_RANK=256 \
F1_CORR_SCALE_INIT=0.10 \
DISTILL_ENABLED=1 \
DISTILL_STEPS=24 \
DISTILL_LR_FACTOR=0.02 \
DISTILL_TEMPERATURE=1.5 \
DISTILL_ALPHA=0.60 \
DISTILL_KL_CLIP=10.0 \
bash concepts/f1/run.sh
```

Speed-first (lighter add-on + shorter distill):

```bash
SEED=1337 \
F1_CORR_RANK=160 \
F1_CORR_SCALE_INIT=0.08 \
DISTILL_ENABLED=1 \
DISTILL_STEPS=12 \
DISTILL_LR_FACTOR=0.015 \
DISTILL_TEMPERATURE=1.3 \
DISTILL_ALPHA=0.50 \
DISTILL_KL_CLIP=8.0 \
bash concepts/f1/run.sh
```
