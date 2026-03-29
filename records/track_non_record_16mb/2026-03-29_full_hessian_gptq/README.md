# Session 05b: Full Hessian GPTQ

**Status**: Smoke-tested, correctness bug found; PR-grounded repair landed, rerun pending
**Parent**: `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor`
**Delta**: Replace naive int6 per-row quantization with Full Hessian GPTQ (Cholesky error compensation)

## Changes vs Anchor

| Change | Anchor | GPTQ Delta |
|--------|--------|------------|
| Quantization | `quantize_int6_per_row` (naive round-to-nearest) | `gptq_quantize_layer` (Cholesky error compensation) |
| Hessian | None | Post-training calibration, H = X^T X, 128 sequences |
| Training code | Unchanged | Unchanged |
| Serialization format | int8 [-32,31] + fp16 per-row scales | Same layout, GPTQ path now uses symmetric int6 clamp `[-31,31]` |
| Dequantization | `dequantize_mixed_int6` | Identical (unchanged) |
| Compression | zstd level 22 | Identical |

## Implemented Algorithm

Current implementation:
1. Collect `H = X^T X` per target linear layer using `forward_logits` on 128 training-sequence samples.
2. Run a PR-grounded GPTQ loop with:
   - actorder permutation
   - Cholesky-based inverse factorization
   - 5-percentile reconstruction search `[0.9990, 0.9995, 0.9999, 0.99999, 1.0]`
   - symmetric int6 clamp `[-31, 31]`
   - PR-style within-block residual update `W_block[:, j:] -= ...`
3. Emit `gptq_layer_diagnostics.json` at export time with:
   - legacy row-max naive MSE
   - percentile-naive MSE
   - GPTQ MSE
   - layer names where GPTQ is worse than either baseline
   - worst block start and max block MSE per layer

## Hyperparameters

- `block_size`: 128
- `percdamp`: 0.01
- `actorder`: True
- `clip_percentiles`: `[0.9990, 0.9995, 0.9999, 0.99999, 1.0]`
- `clip_range`: 31 (clamp to `[-31, 31]`)
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

### 1xH100 smoke (2026-03-29)

Node: `serv-3340`

- stopped at `906` steps in `600202 ms`
- step_avg `662.47 ms`
- pre-quant EMA exact `val_bpb=1.47753094`
- roundtrip exact `val_bpb=1.68963326`
- Hessians collected: `67`
- GPTQ layers used: `66`
- Cholesky fallbacks: `0`
- Hessian collection time: `815 ms`
- GPTQ quantization time: `4236 ms`
- artifact:
  - code `66907` bytes
  - model `7687970` bytes
  - total `7754877` bytes
- job hit the Slurm time limit before sliding eval completed

### Interpretation

- The smoke proves the pipeline mechanics work: calibration, quantization, compression, reload, and eval all run.
- The smoke also proves the current quantizer is wrong: the roundtrip gap is about `0.2121` BPB, far worse than the anchor gap of `0.00775`.
- The training-side metrics from this run are **not comparable** to the `8xH100` anchor because it was a `1xH100` smoke with different `WORLD_SIZE` and `grad_accum_steps`.

### Confirmed divergences from working PR code

- local within-block GPTQ residual propagation used `W_block[:, j + 1:]`, while PRs `#634`, `#1019`, and `#1060` all use `W_block[:, j:]`
- local export used fixed `row_max / 31` scaling only, while the working PRs search 5 clip percentiles and keep the best reconstruction
- local export clamped to `[-32, 31]`, while the working PRs clamp symmetrically to `[-31, 31]`
- local `_classify_param` treated top-level `.proj.` tensors as attention, which pulled an extra `bigram.proj` Hessian
- the local export path had no per-layer naive-vs-GPTQ diagnostics

### 2026-03-29 code repair (not rerun)

- transplanted the GPTQ loop to the PR-grounded structure:
  - same `j:` residual update as `#634/#1019/#1060`
  - 5-percentile search
  - symmetric clamp
- tightened `_classify_param` so only `blocks.*.attn.*` and `blocks.*.mlp.*` are GPTQ targets
- added `gptq_layer_diagnostics.json` and log summaries for layers where GPTQ is worse than:
  - legacy row-max int6
  - percentile-naive int6
- verification in this shell is limited to `py_compile`; no real checkpoint exists in the repo, and this local shell does not have `torch`

### 2026-03-29 replay harness (not rerun from repo shell)

The file now supports export-only replay from an existing checkpoint:

- `EXPORT_ONLY_CHECKPOINT=/path/to/final_model.pt`
- `EXPORT_TAG=...`
- `GPTQ_ACTORDER=0|1`
- `GPTQ_BLOCK_SIZE=...`
- `GPTQ_CALIBRATION_SAMPLES=...`

This is specifically to avoid spending more training time when the export path is still wrong.

### Next move

1. Run export-only replay on the saved checkpoint from the latest smoke.
2. Inspect `gptq_layer_diagnostics*.json`.
3. If GPTQ is still worse than naive on all or most layers, ablate `actorder=False` and `block_size=d_col` on that same checkpoint.
4. Only after the roundtrip gap is sane, rerun `1xH100`, then `8xH100`.
