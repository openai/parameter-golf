# Reproducibility Guide

## Prerequisites

- 4×H100 NVL GPUs (80 GB each) with NCCL support
- RunPod account with API key (or equivalent multi-GPU infrastructure)
- Python 3.10+, PyTorch 2.x with CUDA, flash-attn v3, sentencepiece, brotli, lrzip
- Access to `romeerp/parameter-golf-caseops-v1` on Hugging Face (public dataset)

## Reproducing the Training (exact 6h artifact path)

The PR artifact was **not** produced by a single uninterrupted 360-minute pod. The exact artifact path was a two-pod continuation:

1. **Seed run / downloaded restart point**
   - Local snapshot: `results/8h_longtrain_final/resume_snapshot_step_36452/`
   - Files: `resume_manifest.json` + `resume_rank{0..3}_step36452.pt`
   - Manifest state: `step=36452`, `training_time_ms=18000630.06`, `world_size=4`, `exported_minutes=[60,120,180,240,300]`
   - This is the authoritative 300-minute restart point that was pulled back from the first live pod before it expired.

2. **Resumed 6h-horizon continuation**
   - Output directory: `results/resumed_6h_horizon_continuation_step36452/`
   - Submission artifact: `final_model.int6.360min.ptz`
   - Export metadata: `checkpoint_360min.json` reports `train_steps=49765`, `train_wallclock_seconds=21600.15`, `artifact_bytes=15926271`
   - Continuation log confirms:
     - `schedule_horizon_seconds: 21600.0`
     - `RESUME: restored step=36452, training_time=18000.6s, exported_minutes=[60, 120, 180, 240, 300]`
     - resume saves at 330 min (`step=43125`) and 360 min (`step=49765`)

3. **Later safety snapshot captured during pre-quant follow-up**
   - Local snapshot: `results/prequant_360min_from_step36452/resume_snapshot_step_43062/`
   - Files: `resume_manifest.json` + `resume_rank{0..3}_step43062.pt`
   - Manifest state: `step=43062`, `training_time_ms=19800085.99`, `world_size=4`
   - This was a fallback 330-minute snapshot captured in a separate follow-up pod; it was **not** the artifact-producing continuation run.

### Reproduction requirements

- Resume on **4 GPUs only**. Do not migrate the saved 4-rank snapshot to 8 GPUs.
- Keep the **schedule horizon at 21600s** for the continuation so LR/warmdown semantics remain faithful to the original 6-hour run.
- Reproduce the same two-stage chain: seed run -> download `resume_snapshot_step_36452` -> 4-GPU continuation from that snapshot.
- The later NCCL timeout in the continuation log happened **after** the 360-minute export and 360-minute resume save were written, so it does not affect the submission artifact.

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
  --resume-from results/8h_longtrain_final/resume_snapshot_step_36452 \
  --resume-decompose-only \
  --results-dir results/300min_decompose
```

### 360min pre-quant EMA recovery

```bash
python3 scripts/run_longtrain_scaling.py \
  --num-gpus 4 --max-minutes 90 \
  --resume-from results/8h_longtrain_final/resume_snapshot_step_36452 \
  --prequant-only \
  --max-wallclock 21600 --schedule-horizon 21600 \
  --results-dir results/prequant_360min_from_step36452
```

This pre-quant recovery run also produced a fallback 330-minute snapshot at
`results/prequant_360min_from_step36452/resume_snapshot_step_43062/`.

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
