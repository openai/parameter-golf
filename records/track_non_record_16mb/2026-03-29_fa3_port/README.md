# Session 05 Phase 1: FA3 Port

**Status**: `8xH100` measured, current saved-container FA3 runtime FAILED
**Parent**: `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor`
**Delta**: Replace SDPA attention with direct FA3 (`flash_attn_interface`)

## Changes vs Anchor

| Change | Anchor (SDPA) | FA3 Port |
|--------|--------------|----------|
| Import | `F.scaled_dot_product_attention` | `flash_attn_interface.flash_attn_func` |
| Tensor layout | `(B, H, T, D)` for q/k | `(B, T, H, D)` for all |
| Rotary cache | `(1, 1, T, rd/2)` | `(1, T, 1, rd/2)` |
| q_gain broadcast | `[None, :, None, None]` | `[None, None, :, None]` |
| Post-attention | `.transpose(1, 2).contiguous()` | removed (already B,T,H,D) |
| GQA | `enable_gqa=True` flag | automatic (Hkv < H broadcast) |
| SDPA backend flags | `enable_flash_sdp(True)` etc. | removed |

## Container

Use the saved Pegasus FA3 container, not per-job wheel installs:

- Saved container: `/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh`
- Wheel cache: `/netscratch/$USER/wheels/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl`
- Base image used to build it: NGC `25.02`

Build once:

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

Do not use `--no-deps` against the stock `25.02` image. FA3 import fails with `undefined symbol: aoti_torch_abi_version`.

## Microbenchmark Context

Isolated attention kernel (B=16, T=2048, H=8, Hkv=4, D=64):
- SDPA flash (25.02): 1.889 ms/iter
- FA3 direct (25.02): 0.165 ms/iter (11.44x faster)

The isolated kernel result did not carry over to end-to-end training on the current saved-container runtime.

## Run Commands

Smoke (1xH100):
```bash
srun -p H100 --ntasks=1 --gpus-per-task=1 --cpus-per-task=6 \
  --mem=64G --time=00:10:00 \
  --container-image=/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export PYTHONUNBUFFERED=1 &&
    python records/track_non_record_16mb/2026-03-29_fa3_port/train_gpt.py
  '
```

Full 8xH100 (after smoke passes):
```bash
srun -K -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 \
  --mem=200G --time=00:20:00 \
  --container-image=/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS PYTHONUNBUFFERED=1 &&
    python records/track_non_record_16mb/2026-03-29_fa3_port/train_gpt.py
  '
```

Operational rules:
- Never use `| tail -1` on Pegasus training jobs.
- Keep stdout unbuffered with `PYTHONUNBUFFERED=1` or `python -u`.
- Always force `--nodes=1` for challenge-shaped `8xH100` runs.
- If Slurm allocates multiple nodes for an `8`-GPU job, cancel and relaunch with `--nodes=1`.

## Results

`1xH100` Pegasus smoke:
- Training converged normally through at least step `400`
- Stable `step_avg` after warmup: about `640.2-640.5 ms`
- Loss dropped from `6.9307` to `2.5696` by step `400`
- No FA3 import, numerical, or training-stability failures observed

`8xH100` single-node timing run:
- Node: `serv-3342`
- Step average: `92.67 ms`
- Steps: `6474`
- Sliding s64 exact: `val_bpb=1.12958984`
- Pre-quant EMA exact: `val_bpb=1.14532979`
- Final int6 roundtrip exact: `val_bpb=1.15296145`
- Peak memory: `20825 MiB`
- Total submission size: `15529557` bytes

Comparison vs Session 03 anchor:
- Step average regressed from `91.37 ms` to `92.67 ms` (`+1.30 ms`)
- Steps dropped from `6564` to `6474` (`-90`)
- Sliding s64 regressed from `1.12904446` to `1.12958984` (`+0.00054538`)
- Pre-quant EMA regressed from `1.14472403` to `1.14532979` (`+0.00060576`)
- Roundtrip regressed from `1.15247273` to `1.15296145` (`+0.00048872`)
- Memory improved by `449 MiB`
- Artifact shrank by `221767` bytes

Interpretation:
- This is a clean negative result for the current explicit-FA3 runtime path.
- The attention microbenchmark was not predictive of end-to-end training speed on Pegasus.
- The most likely explanation is runtime-level regression from replacing the vendor-tuned NGC PyTorch stack with the pip-installed generic stack required by the current FA3 wheel.
- Do not rerun this saved-container FA3 path as a throughput candidate.
- If FA3 stays in scope, it must be tested on a vendor-tuned NGC runtime, not the current pip-replaced stack.
