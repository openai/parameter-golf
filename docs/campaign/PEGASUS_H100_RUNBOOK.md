# Pegasus H100 Runbook

Updated: 2026-03-29

## Critical Launcher Lesson

**NEVER use `torchrun --standalone` on Pegasus multi-GPU.**
It hangs at rendezvous. Use Slurm-native `srun` with manual rank mapping.

## Hardware Facts

From Pegasus docs:
- `H100` partition: H100-SXM5, 80GB HBM3, 8 GPUs/node, NVSwitch, **no InfiniBand**
- `H200` partition: H200-SXM5, 141GB HBM3e, 8 GPUs/node, NVSwitch, ~5% faster than H100
- Challenge requires H100 SXM for final verification; H200 is valid for development

## Optimized Run Workflow

### Step 1: Run diagnostic (first time only)

```bash
bash scripts/pegasus_diagnostic.sh
```

This checks: NGC images, /fscratch availability, data, PyTorch version, partition status.

### Step 2: Set up fast data path (first time only)

```bash
bash scripts/pegasus_setup_fscratch.sh
```

Copies training data from `/netscratch` (BeeGFS, higher latency) to `/fscratch` (fast local storage). RunPod uses local NVMe — this closes the I/O gap.

### Step 3: Sync repo

```bash
cd /netscratch/$USER/parameter-golf
git pull origin main
```

### Step 4: Smoke test (single GPU)

```bash
bash scripts/pegasus_smoke_test.sh records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py
```

Runs 90 seconds on 1xA100. Validates: no NaN, EMA works, int6+zstd export, sliding eval.

### Step 5: Full run (8xH100, 600s)

```bash
# Option A: salloc then run
salloc -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 --time=02:00:00
bash scripts/pegasus_optimized_launcher.sh pre_ttt_anchor_8xh100_600s

# Option B: one-shot
bash scripts/pegasus_optimized_launcher.sh pre_ttt_anchor_8xh100_600s \
  records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py --srun
```

The launcher auto-detects:
- NGC container (uses latest available, falls back to bare metal)
- `/fscratch` data (falls back to `/netscratch`)
- CPU thread pinning (MKL/OMP/NUMEXPR all set to 1)
- NCCL tuning (IB disabled, NVSwitch P2P enabled)

## Container Paths

### Standard stable path

- Container: NGC `26.03`
- Use for: normal anchor-derived runs, standard SDPA path, general Pegasus work
- Status: confirmed working end to end

### Explicit FA3 experiment path

- Base container: NGC `25.02`
- Saved container: `/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh`
- Wheel cache: `/netscratch/$USER/wheels/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl`
- Use for: Session 05 FW-1 explicit `flash_attn_interface` experiments
- Status: operationally valid, but the first full `8xH100` run was slower than the SDPA anchor
- Important: do **not** rely on ad hoc per-job wheel installs once the saved container exists. The stock `25.02` container ships an older PyTorch ABI; `--no-deps` wheel installs fail at import time with `undefined symbol: aoti_torch_abi_version`.

Current measured result on Pegasus `8xH100`:
- anchor SDPA: `91.37 ms/step`, sliding s64 `1.12904446`
- saved-container FA3: `92.67 ms/step`, sliding s64 `1.12958984`

Interpretation:
- The current saved-container FA3 path should **not** be rerun as a throughput candidate.
- The likely issue is runtime-level regression from the pip-installed generic PyTorch stack replacing the vendor-tuned NGC build.
- Any future FA3 attempt should target a vendor-tuned NGC runtime, not the current pip-replaced stack.

## Explicit FA3 Bootstrap

Run this once on Pegasus to create the saved FA3 container:

```bash
mkdir -p /netscratch/$USER/wheels /netscratch/$USER/containers
wget -O /netscratch/$USER/wheels/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl \
  "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"

srun -p H100 --ntasks=1 --gpus-per-task=1 --cpus-per-task=6 \
  --mem=64G --time=00:10:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_25.02-py3.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER \
  --container-save=/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh \
  bash -c '
    pip install --no-cache-dir sentencepiece zstandard \
      /netscratch/$USER/wheels/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl &&
    python -c "from flash_attn_interface import flash_attn_func; print(\"FA3 OK\")"
  '
```

Operational rules:
- Keep the exact wheel filename. Do not rename it to a shortened `.whl` name.
- Do not use `--no-deps` on the stock `25.02` container. FA3 imports fail against the bundled PyTorch.
- Do not hide install or training output with `| tail -1`.
- Always set `PYTHONUNBUFFERED=1` or use `python -u` for Pegasus jobs.
- Use the saved container for all subsequent FA3 smoke and full runs.

## Performance Levers (Pegasus vs RunPod gap)

| Lever | How | Expected impact |
|-------|-----|-----------------|
| NGC standard container | `--container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh` | stable default path |
| NGC explicit FA3 path | saved `25.02` FA3 container | benchmark-backed candidate for faster attention |
| /fscratch data | `bash scripts/pegasus_setup_fscratch.sh` | 1-3% faster |
| CPU thread pinning | `MKL_NUM_THREADS=1 OMP_NUM_THREADS=1` | 1-2% faster |
| NCCL NVSwitch tuning | `NCCL_P2P_LEVEL=NVL` | 0-2% faster |
| H200 partition | `-p H200` instead of `-p H100` | ~5% faster (dev only) |

## FA3 Benchmark Commands

These commands benchmark only the isolated attention kernel. They do **not** measure end-to-end training step time.

### Standard SDPA flash path on NGC 26.03

```bash
srun -p H100 --ntasks=1 --gpus-per-task=1 --cpus-per-task=6 \
  --mem=64G --time=00:10:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  python scripts/bench_fa3_vs_sdpa.py --sdpa-only
```

Measured result:
- SDPA flash: `1.967 ms/iter`

### Explicit FA3 path via saved NGC 25.02 container

```bash
srun -p H100 --ntasks=1 --gpus-per-task=1 --cpus-per-task=6 \
  --mem=64G --time=00:10:00 \
  --container-image=/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  python -u scripts/bench_fa3_vs_sdpa.py
```

Measured results:
- SDPA flash: `1.889 ms/iter`
- direct FA3: `0.165 ms/iter`
- relative kernel speedup: `11.44x`

Interpretation:
- This justifies an isolated FA3 training delta.
- It does **not** imply the full training loop will be `11.44x` faster.

## Allocation Shape

```bash
# 8xH100 (challenge parity)
salloc -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 --time=02:00:00

# 8xH200 (development, faster)
salloc -p H200 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 --time=02:00:00

# 1xA100 (smoke test)
salloc -p A100-80GB --ntasks=1 --gpus-per-task=1 --cpus-per-task=6 --time=00:20:00
```

Key rules:
- Use `--ntasks=8 --gpus-per-task=1` (NOT `--gpus=8`)
- Always include `--nodes=1` for challenge-shaped `8xH100` or `8xH200` runs
- Always include `--gpu-bind=none` for NCCL peer-to-peer
- Always include `-K` on srun for kill-on-bad-exit
- If a job lands on multiple nodes for an `8`-GPU run, cancel it and relaunch with `--nodes=1`

## Queue Triage

Use these before waiting on a single-node `8xH100` allocation:

```bash
squeue -p H100 -o "%.8i %.9P %.20j %.8u %.2t %.10M %.10l %.6D %.4C %.8b %R" | head -25
squeue -u $USER --start
```

Interpretation:
- `PD (Resources)` on a `--nodes=1 --ntasks=8` job means no full `8`-GPU node is currently free.
- Partially free GPUs across multiple H100 nodes do not help if the run requires one-node NVSwitch locality.
- Keep the single-node constraint. Do not drop `--nodes=1` just to get an immediate multi-node allocation.

## Environment Variables (always set)

```bash
# Rank mapping (required — Slurm provides these)
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export PYTHONUNBUFFERED=1

# CPU thread pinning (Pegasus known issue: contention)
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export USE_OPENMP=1

# NCCL for NVSwitch-only partition (no InfiniBand on H100)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond,eth
export NCCL_P2P_LEVEL=NVL
```

## Storage

| Path | Use for | Speed | Backup |
|------|---------|-------|--------|
| `$HOME` | Source code only (10GB quota) | Slow | Yes (3h) |
| `/netscratch/$USER` | Repo clone, logs, artifacts | Medium | No |
| `/fscratch/$USER` | Training data (copy from netscratch) | Fast | No |

## Required Run Metadata

Capture for every run:
- GPU model + node hostname
- Exact launch command
- Container image (or "bare-metal")
- Data path (/fscratch or /netscratch)
- step_avg (ms)
- Final post-roundtrip val_bpb
- Sliding window val_bpb (if applicable)
- Artifact bytes (model + code)
- Peak memory

## Custom Container (optional)

If no suitable NGC image exists, build one:

```bash
srun --container-image=/enroot/podman+enroot.sqsh \
  --container-mounts=/dev/fuse:/dev/fuse,/netscratch/$USER:/netscratch/$USER \
  --pty bash

# Inside:
cat > Dockerfile <<'EOF'
FROM nvcr.io/nvidia/pytorch:24.05-py3
RUN pip install --no-cache-dir zstandard sentencepiece huggingface-hub
EOF
podman build . -t pgolf
export ENROOT_SQUASH_OPTIONS="-comp lz4 -Xhc -b 262144"
enroot import -o /netscratch/$USER/pgolf.sqsh podman://pgolf
```

Then use: `--container-image=/netscratch/$USER/pgolf.sqsh`
