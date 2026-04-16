# Colab Experiment: Entropy-Constrained GPTQ Allocator

This is a Colab-oriented April 9 record replica with one intended change: the post-training export allocator.

Baseline for comparison:

- latest valid record: `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`
- mean TTT BPB: `1.0810`
- mean sliding BPB: `1.0827`
- mean artifact bytes: `15,992,694`

The training stack is the April 9 SP8192 3-layer recurrence model: parallel residuals, QK gain `5.25`, EMA, MuonEq-R, and legal score-first TTT. The Colab launcher keeps the same seed family and defaults to seed `42`.

`run.sh` defaults `VAL_LOSS_EVERY=0`, so validation is not consulted during training. The logged BPB metrics are produced after the training loop or after quantized roundtrip export.

## What Changed

`train_gpt.py` now supports `EXPORT_ALLOCATOR=entropy`, which replaces the fixed all-int6 matrix plus int8 embedding export with a compressed-size-aware allocator.

The allocator:

- collects legal GPTQ calibration Hessians from training shards after training
- splits large 2D tensors into stable column groups
- scores each group by Hessian-trace-weighted reconstruction error
- jointly considers per-group matrix bitwidth, per-group clip sigma, embedding bitwidth, and embedding clip sigma
- builds complete candidate artifacts and measures exact compressed bytes
- includes selected code wrapper bytes in the byte constraint
- chooses the lowest reconstruction score candidate under `ARTIFACT_TARGET_BYTES`

The legacy April 9 export remains available with:

```bash
EXPORT_ALLOCATOR=legacy bash run.sh
```

## Colab Usage

```bash
git clone https://github.com/IanniMuliterno/parameter-golf.git
cd parameter-golf/colab/2026-04-14_EntropyConstrained_GPTQ_Allocator
INSTALL_DEPS=1 bash run.sh
```

The SP8192 shards are downloaded from `kevclark/parameter-golf`, matching the April 9 record reproduction. If a Colab runtime has already cached an older manifest, `run.sh` refreshes it before downloading.

On a Colab T4, `run.sh` automatically switches to `COMPUTE_DTYPE=fp16`, disables `torch.compile`, and uses smaller single-GPU batch/calibration defaults. On larger GPUs, use:

```bash
RECORD_PROFILE=1 bash run.sh
```

Set `NPROC_PER_NODE=8` on an 8-GPU box to launch with `torchrun`, matching the April 9 winner's distributed run shape.

## Logs

The main log is written to:

```bash
logs/${RUN_ID}.txt
```

It includes:

- latest valid record comparison target
- pre-quantization post-EMA BPB
- roundtrip quantized BPB
- sliding-window BPB
- legal TTT BPB when `TTT_ENABLED=1`
- artifact bytes and selected code bytes
- training time and eval time

The allocator candidate table is written to:

```bash
logs/allocator_candidates.tsv
```

Columns include lambda, wrapper, code bytes, compressed model bytes, total bytes, Hessian-weighted score, proxy bits, and selected bit-column counts.

## Important Knobs

- `EXPORT_ALLOCATOR=entropy|legacy`
- `ARTIFACT_TARGET_BYTES=16000000`
- `ALLOCATOR_GROUP_COLS=128`
- `ALLOCATOR_MATRIX_BITS=5,6,7`
- `ALLOCATOR_EMBED_BITS=7,8`
- `ALLOCATOR_MATRIX_SIGMAS=10.5,12.85,15.0`
- `ALLOCATOR_EMBED_SIGMAS=16.0,20.0,24.0`
- `ALLOCATOR_LAMBDAS=0,1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5`
- `ALLOCATOR_CODE_WRAPPERS=source,lzma_raw_b85_exec`

For a cheap allocator smoke test:

```bash
ALLOCATOR_MATRIX_BITS=5,6 ALLOCATOR_MATRIX_SIGMAS=12.85 \
ALLOCATOR_EMBED_BITS=7,8 ALLOCATOR_EMBED_SIGMAS=20.0 \
GPTQ_CALIBRATION_BATCHES=4 bash run.sh
```
