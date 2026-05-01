# Reproducibility Guide

## Prerequisites

- 4×H100 NVL GPUs (80 GB each) with NCCL support
- RunPod account with API key (or equivalent multi-GPU infrastructure)
- Python 3.10+, PyTorch 2.x with CUDA, flash-attn v3, sentencepiece, brotli, lrzip
- Access to `romeerp/parameter-golf-caseops-v1` on Hugging Face (public dataset)

## Reproducing the Training (6h continuation)

The 6h artifact was produced by a two-phase training process:

### Phase 1: Initial 4h training

```bash
export RUNPOD_API_KEY=<your-key>
python3 scripts/run_longtrain_scaling.py \
  --num-gpus 4 --max-wallclock 14400 --max-minutes 330 \
  --iterations 200000 --enable-resume \
  --resume-save-minutes "210,240,270,300,330" \
  --export-minutes 240 \
  --train-script records/track_non_record_16mb/2026-04-30_PR1950_LongTrainArtifactScaling/train_gpt.py \
  --results-dir results/4h_longtrain
```

### Phase 2: 6h continuation from 4h checkpoint

```bash
python3 scripts/run_longtrain_scaling.py \
  --num-gpus 4 --continuation-label resumed_6h_horizon \
  --max-wallclock 28800 --schedule-horizon 21600 \
  --export-minutes 360 \
  --resume-save-minutes "330,360,390,420,450,479" \
  --iterations 200000 --max-minutes 720 \
  --resume-from results/4h_longtrain/resume_snapshot_step_36452 \
  --enable-resume \
  --train-script records/track_non_record_16mb/2026-04-30_PR1950_LongTrainArtifactScaling/train_gpt.py \
  --results-dir results/resumed_6h_horizon_continuation_step36452 \
  --run-ttt-sweep-after-train
```

## Reproducing the TTT Sweep (eval-only)

Given the 360-min quantized artifact at `final_model.int6.360min.ptz`:

### Full sweep (all successful variants)

```bash
python3 scripts/run_longtrain_scaling.py \
  --sweep-only-artifact <path-to>/final_model.int6.360min.ptz \
  --ttt-sweep-variants v_sliding_window_control,v0_control_pr1979,v1_rank128_alpha192,v7_noqv_rank96,v12_rank96_phase1_prefix1000 \
  --ttt-max-minutes-per-variant 25 \
  --num-gpus 4 --max-minutes 180 \
  --results-dir results/ttt_sweep_repro
```

### Best variant only (v7_noqv_rank96)

```bash
python3 scripts/run_longtrain_scaling.py \
  --sweep-only-artifact <path-to>/final_model.int6.360min.ptz \
  --ttt-sweep-variants v7_noqv_rank96 \
  --ttt-max-minutes-per-variant 25 \
  --num-gpus 4 --max-minutes 60 \
  --results-dir results/v7_repro
```

### Local execution (no RunPod)

If you have 4×H100 locally:

```bash
python3 scripts/run_longtrain_ttt_sweep.py \
  --artifact <path-to>/final_model.int6.360min.ptz \
  --output-dir ./results/local_sweep \
  --data-path <path-to-caseops-data> \
  --tokenizer-path <path-to-tokenizer.model> \
  --variants v7_noqv_rank96 \
  --ngpus 4 --max-minutes-per-variant 25
```

## Key Environment Variables for v7 (best variant)

```bash
TTT_LORA_RANK=96
TTT_LORA_ALPHA=144
TTT_LORA_LR=0.0001
TTT_BATCH_SIZE=64
TTT_CHUNK_SIZE=48
TTT_K_LORA=1
TTT_MLP_LORA=1
TTT_O_LORA=1
TTT_Q_LORA=0       # Key difference: Q LoRA disabled
TTT_V_LORA=0       # Key difference: V LoRA disabled
GLOBAL_TTT_EPOCHS=1
GLOBAL_TTT_CHUNK_TOKENS=32768
GLOBAL_TTT_BATCH_SEQS=32
PHASED_TTT_PREFIX_DOCS=2000
PHASED_TTT_NUM_PHASES=3
TTT_WARM_START_A=1
TTT_EVAL_ONLY=1
```

## Reproducing Follow-up Controls

### 240min TTT-only control

```bash
python3 scripts/run_longtrain_scaling.py \
  --sweep-only-artifact <path-to>/final_model.int6.240min.ptz \
  --ttt-sweep-variants v0_control_pr1979 \
  --ttt-max-minutes-per-variant 25 \
  --num-gpus 4 --max-minutes 60 \
  --results-dir results/240min_ttt_control
```

### 300min stage decomposition

```bash
python3 scripts/run_longtrain_scaling.py \
  --num-gpus 4 --max-minutes 60 \
  --resume-from <path-to-300min-resume-snapshot> \
  --resume-decompose-only \
  --results-dir results/300min_decompose
```

### 360min pre-quant EMA recovery

```bash
python3 scripts/run_longtrain_scaling.py \
  --num-gpus 4 --max-minutes 90 \
  --resume-from <path-to-330min-resume-snapshot> \
  --prequant-only \
  --max-wallclock 21600 --schedule-horizon 21600 \
  --results-dir results/prequant_360min
```

## Expected Results

| Variant | Expected post_ttt_bpb | Tolerance |
|---------|----------------------|-----------|
| v7_noqv_rank96 | 1.03387 | ±0.0005 (seed/eval variance) |
| v12_rank96_phase1_prefix1000 | 1.03421 | ±0.0005 |
| v0_control_pr1979 | 1.03471 | ±0.0005 |
| sliding_window_control | 1.04273 | ±0.0001 (deterministic) |

## Data Requirements

The CaseOps validation data is downloaded automatically from Hugging Face:
- Repository: `romeerp/parameter-golf-caseops-v1`
- Required files: `fineweb_val_*.bin` shards + tokenizer `.model` file
- Total size: ~200 MB for eval-only

## Hardware Requirements

| Variant | Peak GPU Memory | Wall Time (4×H100) |
|---------|----------------|---------------------|
| v7_noqv_rank96 | 43.6 GiB | ~14 min |
| v0_control_pr1979 | 47.8 GiB | ~15 min |
| v12_rank96_phase1_prefix1000 | 47.7 GiB | ~14 min |
| sliding_window_control | 5.3 GiB | ~2 min |
