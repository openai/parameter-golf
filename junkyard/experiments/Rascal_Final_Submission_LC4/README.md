# RASCAL Final Submission (LC4)

This package is a surgical copy of the submitted Rascal II trainer with one intentional launch delta:

- `COPRIME_MAX_LOADED_SHARDS=4` (loader cache4)

## Integrity

- Source trainer: `experiments/SOTA/2026-03-30_JUNKYARD_RAT_RASCAL_II_nogptq/train_gpt.py`
- Copied trainer: `experiments/Rascal_Final_Submission_LC4/train_gpt.py`
- Status: byte-identical copy (`diff -q` clean)

## Locked race config (via `run.py`)

- `ITERATIONS=20000`
- `WARMDOWN_ITERS=3500`
- `TRAIN_BATCH_TOKENS=786432`
- `MAX_WALLCLOCK_SECONDS=600`
- `SKIP_GPTQ=1`
- `LOADER_MODE=coprime`
- `COPRIME_MAX_LOADED_SHARDS=4`  <-- only upgrade
- `COPRIME_SHARDS_PER_BATCH=1`
- `COPRIME_SHARD_HOLD_STEPS=64`
- `TRIGRAM=0`
- `NGRAM_EVAL_ORDER=0`
- `CUBRIC_CADENCE=0`
- `MTP_NUM_HEADS=0`

## Preflight only (recommended first)

```bash
source /venv/main/bin/activate
python3 experiments/Rascal_Final_Submission_LC4/run.py --mode race --nproc-per-node 8 --seed 444 --dry-run
```

## Full race run

```bash
source /venv/main/bin/activate
python3 experiments/Rascal_Final_Submission_LC4/run.py --mode race --nproc-per-node 8 --seed 444
```

## Notes

- `run.py` defaults to launching with `python -m torch.distributed.run` from the same Python env, avoiding `torchrun` binary mismatch.
- Logs are written under `experiments/Rascal_Final_Submission_LC4/logs/`.
