# TPI-001 Run Notes

## Provenance

- Baseline branch: `baseline/frozen`
- Baseline commit: `5cedb69ac0065e89752f8d3ea1b6990c6833f023`
- Candidate branch: `exp/eval-first-001`
- Candidate implementation commit: `4f6a31ecb45597370cbd4848052ed0f24ca17d0e`

## Candidate command for a real run

```bash
cd /home/eb24516/work/parameter-golf
RUN_ID=tpi001_eval_stride64 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
EVAL_STRIDE=64 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## What was actually executed

- Command: `python3 -m py_compile /home/eb24516/work/parameter-golf/train_gpt.py`
- Result: success

## Missing runtime evidence

- No baseline training run was executed locally.
- No candidate training run was executed locally.
- No `final_model.int8.ptz` was produced locally.

## Blockers

- `torch` is not installed in the current local Python environment.
- `datasets` and `sentencepiece` are not installed in the current local Python environment.
- `data/datasets/fineweb10B_sp1024/` is not present in this workspace.
- `data/tokenizers/fineweb_1024_bpe.model` is not present in this workspace.
- No confirmed 8xH100 runtime is attached to this workspace.
