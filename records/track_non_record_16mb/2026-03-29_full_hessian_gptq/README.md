# Session 05b: Full Hessian GPTQ

**Status**: Implementation complete, awaiting measurement
**Parent**: `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor`
**Delta**: Replace naive int6 per-row quantization with Full Hessian GPTQ (Cholesky error compensation)

## Changes vs Anchor

| Change | Anchor | GPTQ Delta |
|--------|--------|------------|
| Quantization | `quantize_int6_per_row` (naive round-to-nearest) | `gptq_quantize_layer` (Cholesky error compensation) |
| Hessian | None | Post-training calibration, H = X^T X, 128 sequences |
| Training code | Unchanged | Unchanged |
| Serialization format | int8 [-32,31] + fp16 per-row scales | Identical |
| Dequantization | `dequantize_mixed_int6` | Identical (unchanged) |
| Compression | zstd level 22 | Identical |

## Algorithm

Full Hessian GPTQ (Frantar et al., 2023):
1. Collect H = X^T X per linear layer via forward pre-hooks on 128 calibration sequences
2. For each weight matrix, sorted by activation importance (actorder):
   - Cholesky decomposition of H for numerically stable inverse
   - Column-by-column quantization with error compensation
   - Block-wise lazy updates (block_size=128)
3. Same int6 range [-32, 31], same per-row fp16 scales

## Hyperparameters

- `block_size`: 128
- `percdamp`: 0.01
- `actorder`: True
- `clip_range`: 31 (clamp to [-32, 31])
- `calibration_samples`: 128 sequences x 2048 tokens
- `calibration_data`: Training shards (not validation)

## Container

Standard NGC 26.03 (no FA3 dependency):

Smoke (1xH100):
```bash
srun -p H100 --ntasks=1 --gpus-per-task=1 --cpus-per-task=6 \
  --mem=64G --time=00:15:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export PYTHONUNBUFFERED=1
    pip install --no-cache-dir sentencepiece zstandard &&
    python -u records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py
  '
```

Full 8xH100:
```bash
srun -K -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 \
  --mem=200G --time=00:20:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
    export PYTHONUNBUFFERED=1
    export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
    export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond,eth NCCL_P2P_LEVEL=NVL
    pip install --no-cache-dir sentencepiece zstandard &&
    python -u records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py
  '
```

## Success Criteria

- Roundtrip val_bpb strictly < 1.15247273 (anchor)
- Sliding s64 val_bpb < 1.12904446 (anchor)
- Artifact size <= 16,000,000 bytes
- step_avg within +-1ms of 91.37ms (no training impact)
- Zero Cholesky fallbacks

## Results

_Awaiting measurement_
