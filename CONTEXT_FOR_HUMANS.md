# Parameter Golf on DGX Spark: Handoff Document

## TL;DR

We're running OpenAI's "parameter golf" challenge locally on a DGX Spark. The goal is to
train the best small language model (under 16MB) using a single Blackwell GPU. Everything
runs inside Docker with a Streamlit web UI for launching experiments and tracking results.

**Current status**: Working. Training runs at ~8.9 seconds per step with torch.compile enabled.
A Streamlit dashboard is live at http://localhost:8501 with manual runs, run history,
and automated hyperparameter search (Optuna).

## What is Parameter Golf?

OpenAI's challenge: train a language model that fits in 16MB, evaluated by how well it
compresses text (lower bits-per-byte = better). The leaderboard runs are scored on 8xH100
GPUs in 10 minutes. We're using it as a learning/experimentation platform on our single GPU.

**Leaderboard link**: https://github.com/openai/parameter-golf

## Our Hardware

The DGX Spark is a desktop machine with:
- An ARM CPU (not x86 — this matters for Docker images)
- A single Blackwell GPU with 128GB of shared memory
- CUDA 13.0

The GPU is newer than what most software supports out of the box, so we had to find the
right container image (NVIDIA's 25.12 PyTorch container) to get everything working.

## How to Use It

### Starting Everything

```bash
cd ~/parameter-golf
docker compose up -d                    # Start the container
docker compose exec -d parameter-golf streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Then open http://localhost:8501 in your browser.

### The Dashboard (4 Tabs)

1. **Launch Run**: Pick hyperparameters from the sidebar, choose Quick Test (10 min)
   or Overnight Run, and hit Launch. You'll see live training output and a loss chart.

2. **Run History**: Table of all past runs sorted by score (best first). Shows charts
   comparing runs. Click "Best Parameters" to see what settings produced the best result.

3. **Autoresearch**: Automated hyperparameter search using Optuna. Set how many trials
   and how long each should run, then hit Start. It runs in the background — check
   Run History for results as they come in.

4. **System Info**: GPU status, PyTorch version, disk usage, active training processes.

### Running From Command Line

If you prefer the terminal:

```bash
# Quick 10-minute test
docker compose exec parameter-golf bash -c \
  "RUN_ID=my_test ITERATIONS=2000 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=100 bash start_baseline.sh"

# Overnight run (no time limit, 10000 iterations)
docker compose exec parameter-golf bash -c \
  "RUN_ID=overnight_001 ITERATIONS=10000 MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=200 bash start_baseline.sh"

# Automated search (5 trials, 90 min each)
docker compose exec parameter-golf python3 autoresearch.py --trials 5 --minutes-per-trial 90
```

### Stopping

```bash
docker compose down        # Stop everything
docker compose up -d       # Start again (data persists)
```

## What We Tried and What Happened

### Attempt 1: NGC 24.12 + pip PyTorch nightly
- **Problem**: 24.12 container doesn't have Blackwell GPU kernels
- **Fix tried**: Pip-installed PyTorch nightly on top
- **Result**: Basic training worked but torch.compile crashed (gradient shape mismatch bug)
- **Workaround**: Disabled torch.compile, training worked but at 15s/step (slow)

### Attempt 2: NGC 25.12 (current, working)
- **Result**: Native Blackwell support, torch.compile works, 8.9s/step
- **Also fixed**: Flash Attention doesn't have sm_121 kernels, so we use mem_efficient + math
  backends instead (in our custom train_gpt_spark.py)

## Key Numbers to Know

| Metric | Value |
|--------|-------|
| Step time | ~8.9 seconds |
| Steps in 10 min | ~67 |
| Steps in 1 hour | ~400 |
| Steps in 12 hours | ~4800 |
| Upstream baseline score | val_bpb ~1.22 |
| GPU memory used | ~29 GB (of 128 GB shared) |

**Important**: Because we have 1 GPU vs the challenge's 8, our runs take ~8x longer to
reach the same number of steps. A "10-minute" challenge run takes us about 80 minutes.

## Files You Care About

| File | What it does |
|------|-------------|
| `app.py` | The Streamlit web dashboard |
| `train_gpt_spark.py` | Our training script (fork of upstream with Blackwell SDP fix) |
| `start_baseline.sh` | Shell wrapper that downloads data + launches training |
| `run_tracker.py` | Saves/loads experiment results from run_history.json |
| `autoresearch.py` | Headless Optuna hyperparameter search |
| `run_history.json` | All experiment results (auto-created) |
| `logs/` | Training log files (one per run) |
| `Dockerfile` | Container definition (NGC 25.12 base) |
| `docker-compose.yml` | Container config (GPU, ports, volumes) |

## Tunable Hyperparameters

These are the knobs you can turn in the sidebar or via env vars:

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| NUM_LAYERS | 9 | Transformer depth |
| MODEL_DIM | 512 | Hidden size |
| MLP_MULT | 2 | MLP expansion factor |
| MATRIX_LR | 0.04 | Learning rate for weight matrices (Muon optimizer) |
| SCALAR_LR | 0.04 | Learning rate for biases/scales (Adam) |
| TIED_EMBED_LR | 0.05 | Learning rate for tied embeddings |
| MUON_MOMENTUM | 0.95 | Muon optimizer momentum |
| WARMDOWN_ITERS | 1200 | LR warmdown period at end of training |
| ITERATIONS | 20000 | Max training steps |
| MAX_WALLCLOCK_SECONDS | 600 | Time limit (0 = unlimited) |
| SEED | 1337 | Random seed |

## Troubleshooting

**Container won't start**: Run `docker compose logs` to see errors. Usually a GPU driver issue.

**Training crashes immediately**: Check `docker compose exec parameter-golf nvidia-smi` works.
If not, the NVIDIA runtime isn't configured.

**"Not enough SMs" warning**: Normal. The GB10 has fewer streaming multiprocessors than
datacenter GPUs. Training still works fine.

**Streamlit not loading**: Make sure port 8501 isn't blocked. Re-launch with:
`docker compose exec -d parameter-golf streamlit run app.py --server.port 8501 --server.address 0.0.0.0`

**Want to check memory usage**: Don't use nvidia-smi (shows "Not Supported" for memory on UMA).
Instead: `cat /proc/meminfo | grep -i mem`
