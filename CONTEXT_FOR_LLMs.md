# Parameter Golf on DGX Spark: LLM Context File

## What This Project Is

This is a local setup of OpenAI's **parameter-golf** challenge (https://github.com/openai/parameter-golf)
running on an **NVIDIA DGX Spark** desktop workstation. The challenge goal: train the best language
model that fits in a 16MB artifact, evaluated by compression on FineWeb validation (bits-per-byte).

The upstream challenge assumes 8xH100 GPUs. We run on a single Blackwell GB10 GPU.

## Hardware

- **Machine**: NVIDIA DGX Spark
- **CPU**: Grace (ARM64 / aarch64)
- **GPU**: NVIDIA GB10 (Blackwell), compute capability sm_121
- **Memory**: 128 GB unified (shared CPU+GPU) — NOT discrete VRAM
- **Host CUDA**: 13.0, driver 580.126.09
- **sm_120 in PyTorch is binary compatible with sm_121**

## Container Setup

- **Base image**: `nvcr.io/nvidia/pytorch:25.12-py3` (PyTorch 2.10+ with native Blackwell support)
- **Docker Compose**: `docker-compose.yml` with GPU passthrough, ipc=host, port 8501 for Streamlit
- **Volume mount**: `.:/workspace/parameter-golf` — all data, logs, and code persist on host
- **Key env var**: `TORCH_CUDA_ARCH_LIST="12.0"` set in Dockerfile

## Critical Technical Decisions Made

1. **NGC 25.12, NOT 24.12**: The 24.12 container lacks Blackwell sm_120/sm_121 kernels. We tried
   hacking nightly PyTorch on top of 24.12 — it broke torch.compile. The 25.12 container has native
   support and torch.compile works.

2. **SDP backends**: Flash Attention has no sm_121 kernels. In `train_gpt_spark.py` we set:
   - `enable_flash_sdp(False)`
   - `enable_mem_efficient_sdp(True)`
   - `enable_math_sdp(True)`
   Note: In practice, the 25.12 container may actually support flash via a fallback. The training
   log showed flash=True working at 8.9s/step. If you see flash=True in logs, that's OK.

3. **torch.compile WORKS**: Do NOT set TORCH_COMPILE_DISABLE=1. The 25.12 container + torch.compile
   gives ~8.9s/step. Without compile, it was ~15s/step. The "Not enough SMs to use max_autotune_gemm"
   warning is harmless (GB10 has fewer SMs than datacenter GPUs).

4. **Single GPU**: All torchrun commands use `--nproc_per_node=1 --standalone`.

## File Layout

```
parameter-golf/
  Dockerfile                  # NGC 25.12 base, installs extra deps
  docker-compose.yml          # GPU passthrough, ipc=host, port 8501
  requirements.txt            # numpy, torch, streamlit, pandas, optuna, etc.
  train_gpt.py                # Upstream training script (DO NOT MODIFY for experiments)
  train_gpt_spark.py          # Our fork with SDP backend fix for Blackwell
  start_baseline.sh           # Entry point: downloads data if needed, runs torchrun
  app.py                      # Streamlit dashboard (4 tabs: Launch, History, Autoresearch, SysInfo)
  run_tracker.py              # JSON-based run history (save_run, load_runs, parse_log)
  autoresearch.py             # Headless Optuna hyperparameter search CLI
  run_history.json            # Persisted experiment results (created at runtime)
  optuna_study.db             # Optuna SQLite study (created at runtime)
  logs/                       # Training logs: logs/{RUN_ID}.txt
  data/
    datasets/fineweb10B_sp1024/  # Downloaded training/val shards
    tokenizers/                   # BPE tokenizer files
```

## How Hyperparameters Flow

1. Caller sets env vars (e.g., `MATRIX_LR=0.04 ITERATIONS=2000`)
2. `start_baseline.sh` inherits them, sets defaults for DATA_PATH/TOKENIZER_PATH/VOCAB_SIZE
3. `torchrun` launches `train_gpt_spark.py`
4. The `Hyperparameters` class reads everything via `os.environ.get("KEY", default)`

Key tunable params: NUM_LAYERS, MODEL_DIM, NUM_HEADS, NUM_KV_HEADS, MLP_MULT,
MATRIX_LR, SCALAR_LR, TIED_EMBED_LR, EMBED_LR, MUON_MOMENTUM, WARMDOWN_ITERS,
ITERATIONS, MAX_WALLCLOCK_SECONDS, TRAIN_SEQ_LEN, TRAIN_BATCH_TOKENS, SEED.

## Performance Baseline

- **Step time**: ~8.9 seconds (with torch.compile, single GB10)
- **10 min run**: ~67 steps, model barely starts learning
- **1 hour run**: ~400 steps, model starts converging
- **Overnight (12h)**: ~4800 steps, approaches baseline score
- **Baseline score**: val_bpb ~1.22 (from upstream 8xH100 in 10 min)

## Common Operations

```bash
# Start container
cd ~/parameter-golf && docker compose up -d

# Run training
docker compose exec parameter-golf bash -c "RUN_ID=my_run ITERATIONS=2000 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=100 bash start_baseline.sh"

# Launch Streamlit
docker compose exec -d parameter-golf streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Run autoresearch (headless)
docker compose exec parameter-golf python3 autoresearch.py --trials 5 --minutes-per-trial 90

# Check GPU
docker compose exec parameter-golf python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

# View logs
docker compose exec parameter-golf tail -20 logs/my_run.txt
```

## What NOT to Do

- Do NOT use NGC containers older than 25.04 (no Blackwell support)
- Do NOT pip install a different PyTorch version (breaks torch.compile)
- Do NOT set TORCH_COMPILE_DISABLE=1 (halves performance)
- Do NOT use flash_sdp unless you've verified it works in the current container
- Do NOT expect 8xH100 training speeds — single GB10 is ~50x slower per-step
- Do NOT use nvidia-smi for memory monitoring (UMA: "Not Supported"). Use /proc/meminfo
