# Pegasus H100 Runbook

Updated: 2026-03-28

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

## Performance Levers (Pegasus vs RunPod gap)

| Lever | How | Expected impact |
|-------|-----|-----------------|
| NGC container | `--container-image=/enroot/nvcr.io_nvidia_pytorch_24.05-py3.sqsh` | 5-15% faster step_avg |
| /fscratch data | `bash scripts/pegasus_setup_fscratch.sh` | 1-3% faster |
| CPU thread pinning | `MKL_NUM_THREADS=1 OMP_NUM_THREADS=1` | 1-2% faster |
| NCCL NVSwitch tuning | `NCCL_P2P_LEVEL=NVL` | 0-2% faster |
| H200 partition | `-p H200` instead of `-p H100` | ~5% faster (dev only) |

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
- Always include `--gpu-bind=none` for NCCL peer-to-peer
- Always include `-K` on srun for kill-on-bad-exit

## Environment Variables (always set)

```bash
# Rank mapping (required — Slurm provides these)
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

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
