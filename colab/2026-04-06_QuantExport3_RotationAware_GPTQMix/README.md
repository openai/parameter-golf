# Colab Experiment: 2026-04-06_QuantExport3_RotationAware_GPTQMix

This folder is a direct follow-on to [2026-04-06_QuantExport2_selfgenerated_calib](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-04-06_QuantExport2_selfgenerated_calib). It keeps the same Colab launcher shape and benchmark-facing 10-shard setup, but upgrades the post-training export path in three ways:

- rotation-aware quantization
- stronger Hessian approximations from self-generated calibration sequences
- Hessian-driven mixed-precision placement for large tensors

The training model is still the small root-style GPT in [train_gpt.py](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-04-06_QuantExport3_RotationAware_GPTQMix/train_gpt.py). This is not an architecture-faithful replay of the March 25 record, but it is useful for export ablations under a comparable Colab launcher.

## What changed from QuantExport2

`QuantExport2` already had:

- `lzma` export by default
- larger small-tensor float passthrough
- self-generated calibration search over clip percentiles

This folder adds three stronger export ideas.

### 1. Rotation-aware quantization

For large 2D weight matrices, the exporter now tries an orthogonal block rotation before GPTQ-style int8 quantization.

What that does:

- Some matrices have a few large coordinate outliers.
- Int8 quantization is sensitive to those outliers because they consume the dynamic range.
- An orthogonal rotation can spread that energy across dimensions more evenly.
- After quantization, the exporter applies the inverse rotation during dequantization, so the model still sees weights in the original basis.

In this folder, the rotation is a deterministic signed Hadamard-style block rotation chosen from the tensor name. It does not need to store a dense rotation matrix in the artifact.

Why it may help:

- lower reconstruction error for the same int8 payload
- especially useful when a tensor is “spiky” in a few columns

Why it may not help:

- some tensors are already easy to quantize
- on those tensors the exporter keeps the plain non-rotated GPTQ result

### 2. Better Hessian approximations

`QuantExport2` chose clipping percentiles by evaluating candidate exports on self-generated token sequences, but it did not collect layerwise curvature information for the quantizer itself.

This folder now collects an activation second-moment matrix `X^T X` for each large `CastedLinear` weight using self-generated calibration sequences.

What that means in practice:

- the exporter watches the actual input activations flowing into each linear layer
- for each weight matrix, it builds a curvature proxy that says which input directions matter more
- GPTQ-style quantization then uses that proxy when compensating quantization error

This is “better Hessian approximation” in the sense that it is much closer to Hessian-aware GPTQ than plain percentile clipping or plain MSE reconstruction.

Why it may help:

- quantization error is pushed away from directions the model cares about most
- the score used to choose export candidates is less naive than raw weight-space MSE

### 3. Hessian-driven mixed-precision placement

`QuantExport2` had a simple rule:

- keep small tensors or control tensors in float
- quantize the rest

That rule is size-based, not sensitivity-based.

This folder adds a stronger mixed-precision allocator:

- it quantizes large matrices with the Hessian-aware path
- it estimates the Hessian-weighted reconstruction penalty of quantizing each large tensor
- it then spends a small explicit extra-byte budget on the most sensitive large tensors, keeping them in `fp16` instead of int8

So the mixed precision decision is now driven by estimated model damage per byte, not just by tensor size.

Why it may help:

- two tensors of the same size are not equally important
- a few very sensitive matrices can dominate roundtrip loss
- keeping those few tensors in `fp16` can improve `val_loss` and `val_bpb` more than spending the same bytes elsewhere

## How self-generated calibration works here

Self-generated calibration means the trained model creates its own calibration sequences after training.

The exporter:

1. generates autoregressive sequences from the model itself
2. runs those sequences back through the model
3. collects Hessian proxies from the activations
4. builds a compressed-byte-aware candidate grid across clip percentile, rotation on/off, and mixed-precision budget choices
5. quantizes/dequantizes each candidate export
6. scores all candidates on the self-generated sequences and records exact compressed bytes
7. extracts a Pareto frontier in bytes vs calibration loss
8. runs full roundtrip validation on the frontier subset
9. picks the final artifact from the frontier using validation loss under the target-byte budget

No extra training happens during this phase. It is only used to choose a better export.

## Important env vars

Base export knobs:

- `EXPORT_COMPRESSOR`
- `EXPORT_COMPRESSION_LEVEL`
- `INT8_CLIP_PERCENTILE`
- `INT8_KEEP_FLOAT_MAX_NUMEL`
- `INT8_KEEP_FLOAT_STORE_DTYPE`
- `INT8_PER_ROW_SCALE_DTYPE`

Calibration search:

- `SELF_CALIB_VARIANTS`
- `SELF_CALIB_NUM_SEQS`
- `SELF_CALIB_SEQ_LEN`
- `SELF_CALIB_BATCH_SIZE`
- `SELF_CALIB_TEMPERATURE`
- `SELF_CALIB_TOPK`
- `SELF_CALIB_CANDIDATE_PERCENTILES`

New QuantExport3 knobs:

- `GPTQ_BLOCK_SIZE`
  - block size for the GPTQ error-compensation sweep
- `ROTATION_AWARE_ENABLED`
  - `1` to let the exporter try rotated candidates on large matrices
- `ROTATION_BLOCK_SIZE`
  - block width for the signed Hadamard rotation
- `HESSIAN_DAMPING`
  - diagonal damping added before Cholesky-based GPTQ inversion
- `MIXED_PRECISION_EXTRA_BUDGET_BYTES`
  - extra payload budget to spend on keeping sensitive large tensors in `fp16`
- `MIXED_PRECISION_MAX_TENSORS`
  - optional cap on how many large tensors can be upgraded to `fp16`
- `SEARCH_ROTATION_OPTIONS`
  - comma-separated rotation choices for the search loop, typically `0,1`
- `SEARCH_MIXED_PRECISION_BUDGETS`
  - comma-separated byte budgets for the global mixed-precision search
- `SEARCH_MIXED_PRECISION_MAX_TENSORS`
  - comma-separated caps for the global mixed-precision search
- `SEARCH_TARGET_TOTAL_BYTES`
  - total submission-byte target used when selecting the final frontier candidate
- `SEARCH_MAX_FRONTIER_EVALS`
  - maximum number of Pareto-frontier candidates to run on full roundtrip validation

## Default settings in this folder

- `EXPORT_COMPRESSOR=lzma`
- `INT8_CLIP_PERCENTILE=99.99995`
- `INT8_KEEP_FLOAT_MAX_NUMEL=131072`
- `SELF_CALIB_VARIANTS=ar_sample,ar_greedy,ar_topk`
- `SELF_CALIB_NUM_SEQS=24`
- `SELF_CALIB_SEQ_LEN=512`
- `SELF_CALIB_CANDIDATE_PERCENTILES=99.999,99.99984,99.99995,100.0`
- `GPTQ_BLOCK_SIZE=128`
- `ROTATION_AWARE_ENABLED=1`
- `ROTATION_BLOCK_SIZE=128`
- `HESSIAN_DAMPING=0.01`
- `MIXED_PRECISION_EXTRA_BUDGET_BYTES=524288`
- `MIXED_PRECISION_MAX_TENSORS=2`
- `SEARCH_ROTATION_OPTIONS=0,1`
- `SEARCH_MIXED_PRECISION_BUDGETS=0,524288`
- `SEARCH_MIXED_PRECISION_MAX_TENSORS=0,2`
- `SEARCH_TARGET_TOTAL_BYTES=16777216`
- `SEARCH_MAX_FRONTIER_EVALS=6`

These defaults are meant to be stronger than `QuantExport2`, not cheaper. Expect a slower export/search phase.

## Colab usage

```bash
git clone https://github.com/IanniMuliterno/parameter-golf.git
cd parameter-golf/colab/2026-04-06_QuantExport3_RotationAware_GPTQMix
python3 -m pip install -r ../../requirements.txt -r requirements.txt
bash run.sh
```

Or:

```bash
INSTALL_DEPS=1 bash run.sh
```

## Useful overrides

Disable rotation but keep Hessian GPTQ:

```bash
ROTATION_AWARE_ENABLED=0 bash run.sh
```

Turn off the mixed-precision allocator:

```bash
MIXED_PRECISION_EXTRA_BUDGET_BYTES=0 MIXED_PRECISION_MAX_TENSORS=0 bash run.sh
```

Run a cheaper self-calibration pass:

```bash
SELF_CALIB_NUM_SEQS=12 SELF_CALIB_SEQ_LEN=256 bash run.sh
```

Try a more aggressive mixed-precision budget:

```bash
MIXED_PRECISION_EXTRA_BUDGET_BYTES=1048576 MIXED_PRECISION_MAX_TENSORS=4 bash run.sh
```

Search a wider compressed-byte-aware grid:

```bash
SEARCH_MIXED_PRECISION_BUDGETS=0,262144,524288,1048576 SEARCH_MIXED_PRECISION_MAX_TENSORS=0,1,2,4 bash run.sh
```

## Files

- [train_gpt.py](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-04-06_QuantExport3_RotationAware_GPTQMix/train_gpt.py): training plus advanced export logic
- [run.sh](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-04-06_QuantExport3_RotationAware_GPTQMix/run.sh): Colab launcher with benchmark-aligned data/view defaults
- [requirements.txt](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-04-06_QuantExport3_RotationAware_GPTQMix/requirements.txt): extra Python dependencies

## Outputs

Run from this folder and the script emits:

- `logs/`
- `final_model.pt`
- `final_model.int8.ptz` or `final_model.int8.ptx`
- `logs/artifact_size_table.tsv`
- `logs/search_frontier.tsv`
- `logs/roundtrip_quality_table.tsv`
- `runtime_data/fineweb10B_sp1024_10shards/`
