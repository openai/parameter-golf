# Experiment Workflow

This runbook is for participants who want a reproducible path from local Apple Silicon smoke tests to leaderboard-oriented CUDA sweeps.

It deliberately does not change `train_gpt.py`. The goal is to keep the baseline trainer intact while adding repeatable experiment hygiene around it.

## Scope

- Local Apple Silicon is for fast smoke tests and data/tokenizer validation.
- Remote CUDA runs are for actual leaderboard-oriented sweeps.
- Wave 1 keeps tokenizer and dataset fixed and focuses on post-quantization `val_bpb`.

## Verified Local Setup

The following sequence was verified on an Apple Silicon Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

Observed smoke result from `logs/mlx_smoke.txt`:

- `step:200/200 train_loss:3.8990`
- `train_time:92686ms`
- `step_avg:463.43ms`
- `tok_s:17698`

That run is useful as a trainer/data sanity check. It is not a good ranking loop because final validation still runs on the full fixed validation split.

## Wave 1 Strategy

Use cheap breadth runs first, then spend 8xH100 only on confirmation.

### Fixed Contract

Hold these constant for the first wave:

- `DATA_PATH=<repo>/data/datasets/fineweb10B_sp1024`
- `TOKENIZER_PATH=<repo>/data/tokenizers/fineweb_1024_bpe.model`
- `VOCAB_SIZE=1024`
- `NUM_LAYERS=9`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=2`
- `TIE_EMBEDDINGS=1`
- `TRAIN_SEQ_LEN=1024`
- `TRAIN_BATCH_TOKENS=524288`
- `VAL_BATCH_SIZE=524288`
- `SEED=1337`

### Ranking Metric

Rank experiments by:

1. `final_int8_zlib_roundtrip_exact val_bpb`
2. `Total submission size int8+zlib`
3. steady-state `step_avg`

Do not rank by pre-quantization validation loss alone.

### Stage A: 1xH100 Breadth Screen

Recommended defaults:

```bash
MAX_WALLCLOCK_SECONDS=300
VAL_LOSS_EVERY=0
TRAIN_LOG_EVERY=200
ITERATIONS=20000
WARMUP_STEPS=20
WARMDOWN_ITERS=1200
TIED_EMBED_LR=0.05
MATRIX_LR=0.04
SCALAR_LR=0.04
MUON_MOMENTUM=0.95
MUON_BACKEND_STEPS=5
MUON_MOMENTUM_WARMUP_START=0.85
MUON_MOMENTUM_WARMUP_STEPS=500
BETA1=0.9
BETA2=0.95
ADAM_EPS=1e-8
GRAD_CLIP_NORM=0.0
QK_GAIN_INIT=1.5
LOGIT_SOFTCAP=30
ROPE_BASE=10000
```

Recommended first-pass sweep order:

1. `TIED_EMBED_LR`
2. `MATRIX_LR`
3. `SCALAR_LR`
4. `MUON_MOMENTUM`
5. `QK_GAIN_INIT`
6. `LOGIT_SOFTCAP`

Do not open tokenizer changes, untied embeddings, or larger width/depth in wave 1.

### Stage B: 1xH100 Confirm

Re-run the best 2-3 configs at a longer equal-token budget with periodic validation enabled. This is where seed variance checks start.

### Stage C: 8xH100 Confirm

Only promote the best candidates to `nproc_per_node=8` and the 600-second track cap.

## Run Isolation

`train_gpt.py` writes:

- `final_model.pt`
- `final_model.int8.ptz`
- `logs/<RUN_ID>.txt`

For sweeps, every run should get its own working directory. The helper script in `scripts/run_wave1_screen.sh` creates:

- `runs/<date>/<run_id>/command.sh`
- `runs/<date>/<run_id>/env.txt`
- `runs/<date>/<run_id>/train.log`
- `runs/<date>/<run_id>/logs/<run_id>.txt`
- `runs/<date>/<run_id>/final_model.pt`
- `runs/<date>/<run_id>/final_model.int8.ptz`

## Ledger Fields

Track at least:

- `run_id`
- `git_sha`
- `world_size`
- `seed`
- `data_path`
- `tokenizer_path`
- `train_shards`
- `vocab_size`
- `num_layers`
- `model_dim`
- `num_heads`
- `num_kv_heads`
- `mlp_mult`
- `tie_embeddings`
- `train_batch_tokens`
- `val_batch_size`
- `train_seq_len`
- `iterations`
- `warmup_steps`
- `warmdown_iters`
- `max_wallclock_seconds`
- `val_loss_every`
- `tied_embed_lr`
- `matrix_lr`
- `scalar_lr`
- `muon_momentum`
- `muon_backend_steps`
- `qk_gain_init`
- `logit_softcap`
- `step_stop`
- `step_avg_ms`
- `peak_mem_allocated_mib`
- `peak_mem_reserved_mib`
- `final_int8_zlib_roundtrip_exact_val_bpb`
- `bytes_total_int8_zlib`

The parser at `scripts/extract_run_metrics.py` emits one JSON object per run using the stable log lines already printed by `train_gpt.py`.

## Remote Usage

From the repository root on a CUDA machine:

```bash
bash scripts/run_wave1_screen.sh --dry-run
bash scripts/run_wave1_screen.sh
python3 scripts/extract_run_metrics.py runs/*/*/logs/*.txt
```

Use `--dry-run` first so the full command matrix is visible before the runs start.
