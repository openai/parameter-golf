# Remote Runbook

This repo is ready for the CUDA path.

## Recommended Path

Use the official Runpod Parameter Golf template mentioned in [README.md](/Users/deividasmataciunas/Desktop/research/openai_golf/README.md).

Start with one of these:

- `1x H100`: cheapest realistic sanity-check path for code, logs, artifact size, and eval behavior.
- `8x H100 SXM`: record-track run once the recipe looks stable.

## First-Time Remote Setup

On the remote box:

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
git remote add myfork <your-fork-url>
git fetch myfork
git checkout <your-branch-with-our-changes>
```

Then hydrate the published cache:

```bash
TRAIN_SHARDS=1 bash scripts/remote_fetch_data.sh
```

For a fuller training prefix:

```bash
TRAIN_SHARDS=10 bash scripts/remote_fetch_data.sh
```

## First Experiment

This is the first recipe to run against our merged script:

```bash
NPROC_PER_NODE=1 bash scripts/run_remote_experiment.sh
```

For a full multi-GPU run:

```bash
NPROC_PER_NODE=8 bash scripts/run_remote_experiment.sh
```

## What This Recipe Uses

- `10` layers
- fp16 tied-embedding export
- NTK-aware longer eval support
- sliding-window eval with stride `64`
- decoupled Muon weight decay
- overtone embedding init
- phase-shaped residual mixing init

## First Ablations To Queue

Run these one at a time after the first successful remote run:

```bash
EVAL_STRIDE=0 NPROC_PER_NODE=8 bash scripts/run_remote_experiment.sh
EVAL_SEQ_LEN=2048 NPROC_PER_NODE=8 bash scripts/run_remote_experiment.sh
NUM_LAYERS=9 NPROC_PER_NODE=8 bash scripts/run_remote_experiment.sh
MUON_WEIGHT_DECAY=0.00 NPROC_PER_NODE=8 bash scripts/run_remote_experiment.sh
OVERTONE_INIT_POWER=0.00 NPROC_PER_NODE=8 bash scripts/run_remote_experiment.sh
```

## What To Look For

- `step_avg`
- final `val_bpb`
- final `final_int8_zlib_roundtrip_exact`
- final `final_int8_ttt_lora`
- total `int8+zlib` artifact bytes

If you send me a remote log, I can turn it into the next ablation decision quickly.
