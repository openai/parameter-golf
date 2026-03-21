# Parameter Golf Cloud Worker Runbook

## Purpose
This file is the operational runbook for remote experimentation on a cloud GPU worker.

Use this alongside `program.md`:
- `program.md` defines research policy and experiment priorities
- `RUNBOOK.md` defines worker setup, lifecycle, and standard commands

## Default Worker Shape
- `1x RTX 5090`
- SSH enabled
- persistent volume preferred
- repo mounted or cloned under `/workspace/parameter-golf`

## Required Remote Layout
- repo path: `/workspace/parameter-golf`
- venv path: `/workspace/parameter-golf/.venv`
- logs directory: `/workspace/parameter-golf/logs`
- data path: `/workspace/parameter-golf/data/datasets/fineweb10B_sp1024`
- tokenizer path: `/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`

## Worker Bootstrap
Run these on a fresh worker:

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
mkdir -p logs
```

If the repository already exists:

```bash
cd /workspace/parameter-golf
. .venv/bin/activate
git status --short
```

## Syncing Current Local Trainer
If the local controller has the intended `train_gpt.py`, copy it to the worker before a run:

```bash
scp -i ~/.ssh/runpod_parameter_golf_ed25519 -P <PORT> \
  ./train_gpt.py root@<HOST>:/workspace/parameter-golf/train_gpt.py
```

## Smoke Run
Use this to verify the path works end to end:

```bash
cd /workspace/parameter-golf
. .venv/bin/activate
RUN_ID=smoke_001 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=60 \
TRAIN_BATCH_TOKENS=131072 \
VAL_BATCH_SIZE=131072 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=20 \
python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py
```

## Search Run
Use this for ordinary hypothesis testing:

```bash
cd /workspace/parameter-golf
. .venv/bin/activate
RUN_ID=search_001 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=300 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100 \
python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py
```

## Confirm Run
Use this only for promising candidates:

```bash
cd /workspace/parameter-golf
. .venv/bin/activate
RUN_ID=confirm_001 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=100 \
python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py
```

## Metrics To Parse
Every completed run should extract:
- `final_int8_zlib_roundtrip_exact`
- `Total submission size int8+zlib`
- `stopping_early: wallclock_cap`
- `peak memory allocated`

These should be copied into the experiment ledger after every run.

## Quick Log Checks
Useful commands:

```bash
cd /workspace/parameter-golf
rg "final_int8_zlib_roundtrip_exact|Total submission size int8\\+zlib|stopping_early" logs
```

```bash
cd /workspace/parameter-golf
ls logs
```

## Worker Lifecycle
Preferred pattern:
1. Start pod
2. SSH in
3. Run one or more experiments
4. Sync or inspect results
5. Stop pod when idle

Do not leave the worker running when not actively experimenting.

## Notes
- Prefer a persistent volume for repeated sessions.
- Do not change dataset or tokenizer during the initial search phase.
- Keep the worker simple: one GPU, one repo, one active branch of experiments.
