# Session 05c-plus: Training Bundle on Session 03 Anchor

## Changes from anchor

| # | Change | Type | Lines changed |
|---|--------|------|---------------|
| 1 | warmdown 3000 -> 3500 | constant | 1 |
| 2 | XSA 4 -> 11 (all layers) | constant | 1 |
| 3 | LeakyReLU(0.5)^2 (replaces ReLU^2) | activation | 1 |
| 4 | VE128 on layers 9-10 | new module | ~50 |

## Base

`records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py`

## NOT included

- SWA (dead code in PR #1019 and #634 — collected but never applied)
- GPTQ (parked after 7 ablations on current anchor)
- FA3 (container ABI issue unresolved)

## Success criteria

- Sliding s64 val_bpb < 1.1260 (anchor: 1.1290)
- Pre-quant EMA val_bpb < 1.1420 (anchor: 1.14472)
- step_avg within +5ms of anchor (91.37 ms)
- Artifact <= 16,000,000 bytes

## Launch

```bash
cd /netscratch/$USER/parameter-golf && git pull

srun -K -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 \
  --mem=200G --time=00:20:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
    export PYTHONUNBUFFERED=1
    export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
    export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond,eth NCCL_P2P_LEVEL=NVL
    pip install --no-cache-dir sentencepiece zstandard 2>/dev/null
    python -u records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py
  '
```
