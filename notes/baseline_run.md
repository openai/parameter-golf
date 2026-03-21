# Baseline Run Memo

## Current frozen baseline

- Branch: `baseline/frozen`
- Baseline code SHA: `45bbccff356439d2f0b0dbae06cc3fa58b9576ed`
- Policy: keep the model/training code identical to the published baseline until the first TPI-driven experiment branch is opened.

## Leaderboard-relevant baseline command candidate

This is the closest command to the repository baseline described in the root README.

```bash
cd /home/eb24516/work/parameter-golf
mkdir -p runs/baseline/baseline_sp1024
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected outputs

- Script-managed log: `logs/${RUN_ID}.txt`
- Script-managed artifacts in repo root: `final_model.pt`, `final_model.int8.ptz`
- Local run archive target for this workspace: `runs/baseline/<run_id>/`
- Suggested post-run archive contents: copied `logs/${RUN_ID}.txt`, captured stdout/stderr, selected metrics summary, regenerated size report

## Run template

- Branch: `baseline/frozen`
- Commit SHA: `45bbccff356439d2f0b0dbae06cc3fa58b9576ed`
- Seed: `1337` by default (`SEED` env var in `train_gpt.py`)
- Train time target: `600s` for the leaderboard path
- Primary validation metric: final `val_bpb` from `final_int8_zlib_roundtrip_exact`
- Output destination: `runs/baseline/<run_id>/` plus script-native `logs/`

## Execution status

- Full baseline execution in this local workspace: not run
- Executable now: no

## Current blockers

- `torch` is not installed in the current Python environment.
- `datasets` and `sentencepiece` are not installed in the current Python environment.
- `data/datasets/fineweb10B_sp1024/` is not present.
- `data/tokenizers/fineweb_1024_bpe.model` is not present.
- No confirmed 8xH100 runtime is attached to this workspace.

## Notes

- If a smaller smoke run is needed before remote execution, prefer a separate branch or a clearly labeled `RUN_ID` so that the frozen baseline provenance stays clean.
- Docs-only commits on `baseline/frozen` are acceptable; they do not change `train_gpt.py` behavior.
