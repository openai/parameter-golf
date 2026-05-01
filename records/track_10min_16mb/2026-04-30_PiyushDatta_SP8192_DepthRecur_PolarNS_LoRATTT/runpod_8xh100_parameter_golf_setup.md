# RunPod 8×H100 setup for `parameter-golf-fork`

This is a clean setup/runbook for a **fresh RunPod 8×H100 machine**:

-   do **not** rely on the small root filesystem for datasets/logs
-   keep the **repo + dataset/cache/logs on `/dev/shm`** for speed and to avoid quota/root-disk issues
-   keep the **uv environment on `/workspace`**, not `/dev/shm`, so PyTorch shared libraries load correctly
-   disable Hugging Face Xet downloads with `HF_HUB_DISABLE_XET=1`

---

## 1) Check the machine and storage

```
nvidia-smi && df -h
```

## 2) Clone the repo into `/dev/shm`

```bash
cd /dev/shm
git clone https://github.com/PiyushDatta/parameter-golf-fork.git
cd /dev/shm/parameter-golf-fork
```

---

## 3) Create a uv environment on `/workspace`

```
export UV_LINK_MODE=copy
export UV_CACHE_DIR=/dev/shm/uv-cache
mkdir -p /dev/shm/uv-cache
uv venv /workspace/uv-envs/parameter-golf
source /workspace/uv-envs/parameter-golf/bin/activate
cd /dev/shm/parameter-golf-fork
uv sync --active
uv sync --active --reinstall-package torch
```

## 4) Verify PyTorch if needed

```
/workspace/uv-envs/parameter-golf/bin/python -c "import torch; print(torch.__file__); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); import torch.distributed.run; print('ok')"
```

---

## 5) Configure Hugging Face cache + temp dirs in `/dev/shm`

```
export HF_HUB_DISABLE_XET=1
export HF_HOME=/dev/shm/hf-cache
export HUGGINGFACE_HUB_CACHE=/dev/shm/hf-cache/hub
export HF_DATASETS_CACHE=/dev/shm/hf-cache/datasets
export TMPDIR=/dev/shm
mkdir -p /dev/shm/hf-cache/hub
mkdir -p /dev/shm/hf-cache/datasets
mkdir -p /dev/shm/pg-logs
```

---

## 6) Download the dataset

```
cd /dev/shm/parameter-golf-fork
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
ls -lah data/datasets/fineweb10B_sp8192 | head
ls -lah data/tokenizers | grep 8192
```

You should see files like:

-   `data/datasets/fineweb10B_sp8192/fineweb_train_000000.bin`
-   `data/tokenizers/fineweb_8192_bpe.model`

---

## 7) Launch training

Use the **venv Python** explicitly.

```
cd /dev/shm/parameter-golf-fork

mkdir -p /workspace/pg-tmp
mkdir -p /workspace/torchinductor-cache
mkdir -p /workspace/triton-cache

export HF_HUB_DISABLE_XET=1
export HF_HOME=/dev/shm/hf-cache
export HUGGINGFACE_HUB_CACHE=/dev/shm/hf-cache/hub
export HF_DATASETS_CACHE=/dev/shm/hf-cache/datasets
export TMPDIR=/workspace/pg-tmp
export TORCHINDUCTOR_CACHE_DIR=/workspace/torchinductor-cache
export TRITON_CACHE_DIR=/workspace/triton-cache

rm -rf /dev/shm/torchinductor_root /dev/shm/triton

bash records/track_10min_16mb/2026-04-30_PiyushDatta_SP8192_DepthRecur_PolarNS_LoRATTT/run_final_submission.sh --nproc 8
```

The wrapper already saves the full console stream for each seed under:

```text
records/track_10min_16mb/2026-04-30_PiyushDatta_SP8192_DepthRecur_PolarNS_LoRATTT/logs/seed_<seed>.log
```

The underlying training script also writes its own per-run log under `./logs/` and prints the exact path near the top as:

```text
logfile: logs/<run_id>.txt
```

---

## 8) View logs

`git add -f records/track_10min_16mb/2026-04-30_PiyushDatta_SP8192_DepthRecur_PolarNS_LoRATTT/logs/`

`git diff --staged`

## 9) (Optional) Save logs/results somewhere persistent

Because `/dev/shm` is temporary, copy logs back out after the run.

This wrapper normally runs a seed sweep, so treat the per-seed logs as the primary artifacts.

The safest pattern is:

1. Copy every wrapper per-seed console log from `records/.../logs/seed_<seed>.log`
2. Optionally copy the underlying per-run script logs from `./logs/<run_id>.txt`
3. Copy each seed log into the record folder as `train_seed<seed>.log`
4. Pick one canonical seed log and copy it to `train.log`

```bash
mkdir -p /workspace/pg-results

RECORD_DIR=/dev/shm/parameter-golf-fork/records/track_10min_16mb/2026-04-30_PiyushDatta_SP8192_DepthRecur_PolarNS_LoRATTT
SEED_LOG_DIR="$RECORD_DIR/logs"
RUN_LOG_DIR=/dev/shm/parameter-golf-fork/logs
SEEDS=(42 314 999)
CANONICAL_SEED=42

# Copy the wrapper's per-seed logs to persistent storage.
cp "$SEED_LOG_DIR"/seed_*.log /workspace/pg-results/ 2>/dev/null || true

# The training script also prints "logfile: logs/<run_id>.txt". Copy those too if needed.
cp "$RUN_LOG_DIR"/*.txt /workspace/pg-results/ 2>/dev/null || true

# Update the record directory seed logs.
for SEED in "${SEEDS[@]}"; do
  cp "$SEED_LOG_DIR/seed_${SEED}.log" "$RECORD_DIR/train_seed${SEED}.log"
done

# Pick one canonical seed log for train.log.
cp "$SEED_LOG_DIR/seed_${CANONICAL_SEED}.log" "$RECORD_DIR/train.log"
```

If the run writes any result files or artifacts, copy those too.
