# Colab Experiment: 2026-04-06_QuantExport_Improvements_Benchmark

This folder provides a Google Colab launcher for testing quantization/export changes on top of the root [train_gpt.py](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/train_gpt.py).

The benchmark reference is [colab/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/run.sh](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/run.sh). This experiment keeps the launcher-side comparison surface aligned with that run where the root trainer supports it, while isolating post-training export changes.

## What this experiment is for

- Use the root trainer as the reference code path.
- Run on Google Colab with one GPU.
- Train against a pinned 10-shard FineWeb `sp1024` view.
- Test export-only knobs without mixing in dataset-view drift or launcher drift.
- Emit explicit `training_step:<n>/<iterations>` lines in the log in addition to warmup progress.

## What was done to make results comparable to the March 25 benchmark

- The launcher uses the same dataset family: `data/datasets/fineweb10B_sp1024`.
- It creates the same style of local 10-shard training view and reuses the same validation shard files.
- It uses the same tokenizer family and path shape: `fineweb_1024_bpe.model`.
- It keeps the same single-run control defaults that matter for wallclock comparisons:
  - `TRAIN_SHARDS=10`
  - `TRAIN_BATCH_TOKENS=65536`
  - `VAL_BATCH_SIZE=65536`
  - `TRAIN_SEQ_LEN=2048`
  - `ITERATIONS=20000`
  - `WARMUP_STEPS=20`
  - `WARMDOWN_ITERS=4000`
  - `MAX_WALLCLOCK_SECONDS=600`
  - `VAL_LOSS_EVERY=4000`
  - `TRAIN_LOG_EVERY=200`
  - `SEED=314`
- It uses the same Colab-friendly dtype fallback policy:
  - `bf16` on GPUs that support CUDA bf16
  - `fp16` otherwise
- It keeps the benchmark’s practical single-GPU stance by disabling compile and fused Adam by default unless you override them.

## Important non-comparable parts

This is not an architecture-faithful replay of the March 25 record.

- The March 25 benchmark folder runs the record script from [records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py).
- This folder runs it's own [train_gpt.py](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-04-06_QuantExport2_selfgenerated_calib/train_gpt.py), which is a smaller baseline trainer, similar to what we have in the root.
- Because of that, absolute `val_bpb` numbers are still useful for directional benchmarking, but they are not an apples-to-apples measure of the March 25 record architecture.
- We can although, make an full apple-to-apple comparison with other `QuantExport` scripts.

In short: this folder is comparable on dataset view, tokenizer, wallclock budget, shard count, seed, and logging cadence. It is not comparable on model architecture or legal-record export stack.

## Export knobs being tested

The root trainer now exposes export settings through env vars. This folder defaults to an `lzma` export so you can test export improvements directly.

It now also supports self-generated calibration search for export tuning. The April 6 launcher enables a small calibration search by default, using several self-generated decoding variants to choose the int8 clipping percentile before writing the final artifact.

- `EXPORT_COMPRESSOR`
  - `zlib` or `lzma`
- `EXPORT_COMPRESSION_LEVEL`
  - `zlib`: typical range `0-9`
  - `lzma`: typical range `0-9`
- `EXPORT_ARTIFACT_NAME`
  - default is chosen from the compressor
- `INT8_CLIP_PERCENTILE`
  - clipping percentile before int8 quantization
- `SELF_CALIB_VARIANTS`
  - comma-separated self-generated calibration modes
  - supported values: `ar_sample`, `ar_greedy`, `ar_topk`, `ar_mixed`
- `SELF_CALIB_NUM_SEQS`
  - number of self-generated calibration sequences
- `SELF_CALIB_SEQ_LEN`
  - length of each self-generated calibration sequence
- `SELF_CALIB_BATCH_SIZE`
  - generation/eval batch size for the calibration search
- `SELF_CALIB_TEMPERATURE`
  - sampling temperature for non-greedy calibration variants
- `SELF_CALIB_TOPK`
  - top-k value used by `ar_topk`
- `SELF_CALIB_CANDIDATE_PERCENTILES`
  - comma-separated list of clipping percentiles to score on calibration sequences
- `INT8_KEEP_FLOAT_MAX_NUMEL`
  - tensors at or below this size stay in float form
- `INT8_KEEP_FLOAT_STORE_DTYPE`
  - float storage dtype for kept tensors
- `INT8_PER_ROW_SCALE_DTYPE`
  - dtype used for saved per-row quant scales

The default settings in this folder are:

- `EXPORT_COMPRESSOR=lzma`
- `EXPORT_COMPRESSION_LEVEL=9`
- `INT8_CLIP_PERCENTILE=99.99984`
- `SELF_CALIB_VARIANTS=ar_sample,ar_greedy,ar_topk`
- `SELF_CALIB_NUM_SEQS=12`
- `SELF_CALIB_SEQ_LEN=256`
- `SELF_CALIB_CANDIDATE_PERCENTILES=99.99,99.995,99.999,99.99984,100.0`
- `INT8_KEEP_FLOAT_MAX_NUMEL=65536`
- `INT8_KEEP_FLOAT_STORE_DTYPE=fp16`
- `INT8_PER_ROW_SCALE_DTYPE=fp16`

## Logging

The root trainer now logs:

- `warmup_step:<n>/<warmup_steps>`
- `training_step:<n>/<iterations>`

That gives you explicit progress for both warmup and main training.

## Files

- [train_gpt.py](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-04-06_QuantExport_Improvements_Benchmark/train_gpt.py): thin entrypoint into the root trainer.
- [run.sh](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-04-06_QuantExport_Improvements_Benchmark/run.sh): prepares the 10-shard view, sets benchmark-aligned env vars, and launches training.
- [requirements.txt](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-04-06_QuantExport_Improvements_Benchmark/requirements.txt): extra Colab dependencies for this folder.

## Colab usage

```bash
git clone https://github.com/IanniMuliterno/parameter-golf.git
cd parameter-golf/colab/2026-04-06_QuantExport_Improvements_Benchmark
python3 -m pip install -r ../../requirements.txt -r requirements.txt
bash run.sh
```

Or let the launcher install dependencies:

```bash
INSTALL_DEPS=1 bash run.sh
```

## Useful overrides

Baseline-style export:

```bash
EXPORT_COMPRESSOR=zlib EXPORT_COMPRESSION_LEVEL=9 bash run.sh
```

More aggressive small-tensor passthrough:

```bash
INT8_KEEP_FLOAT_MAX_NUMEL=131072 bash run.sh
```

Different clipping percentile:

```bash
INT8_CLIP_PERCENTILE=99.9990 bash run.sh
```

Disable self-generated calibration search:

```bash
SELF_CALIB_VARIANTS= SELF_CALIB_NUM_SEQS=0 bash run.sh
```

Try a different calibration mix:

```bash
SELF_CALIB_VARIANTS=ar_sample,ar_greedy,ar_mixed SELF_CALIB_NUM_SEQS=18 bash run.sh
```

## Outputs

Run from this folder and the root trainer will emit outputs here:

- `logs/`
- `final_model.pt`
- `final_model.int8.ptz` or `final_model.int8.ptx`
- `runtime_data/fineweb10B_sp1024_10shards/`
