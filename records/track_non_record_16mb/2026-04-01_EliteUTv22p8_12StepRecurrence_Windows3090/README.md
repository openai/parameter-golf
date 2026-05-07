# Parameter Golf — Architecture + Research Dossier (Windows RTX 3090)

This repository contains the current training stack and research history for our Parameter Golf system. We started from a Universal Transformer + depth recurrence idea, then iteratively evolved into a different architecture/training regime driven by wallclock-constrained BPB results.

## Executive Summary

- **Platform**: Windows 11 + RTX 3090
- **Constraint**: 10-minute wallclock runs
- **Main journey** (source: `logs/bpb_full_journey.csv`):
  - ~**3.75 BPB** (early 12-step UT)
  - ~**3.18 BPB** (stabilized UT + tuning)
  - ~**2.73 BPB** (throughput/shape pivot)
  - ~**1.81 BPB** (new architecture regime)

## What the current code implements

From `model.py` + `train_gpt.py`, the current system is best described as:

> **A tied-block recurrent causal LM backbone with per-step adapters/control paths and architecture-aware optimizer routing.**

Key characteristics:

1. **Single recurrent block reused across depth** (`num_steps`, tied weights)
2. **Step-conditioned behavior** via:
   - per-step embeddings
   - per-step LoRA adapters (`RelaxedLinear`)
   - per-step value bias (`v_step_bias`)
   - optional level signal, smeargate, bigram hash
3. **Causal decoder objective** with final softcapped logits
4. **Training stack tightly coupled to architecture**:
   - Muon for dense matrices
   - AdamW groups for scalar/control/LoRA/embeddings
   - recurrence-aware gradient scaling
   - wallclock-aware schedule, EMA, and stability instrumentation

### Current architecture snapshot (active deterministic re-eval profile)

This is the current working architecture/profile used for deterministic reevaluation via `ScaleDown.bat`:

- `MODEL_DIM=1024`, `NUM_HEADS=8`, `NUM_KV_HEADS=4`
- `RECURRENCE_STEPS=1` (single-step tied-block path)
- `MLP_MULT=5`
- `LORA_RANK=512`, `LORA_SCOPE=q`
- Feature toggles:
  - `SMEARGATE_ENABLED=0`
  - `BIGRAM_HASH_ENABLED=1` (`SIZE=2048`, `SCALE=0.05`)
  - `LEVEL_SIGNAL_ENABLED=0`
  - `ORTHO_INIT=0`

## BPB Journey (High-level)

### Phase 1 — Original UT depth recurrence era

| Milestone | Val BPB | Notes |
|---|---:|---|
| `Initial_UT_12step` | 3.75 | First stable 12-step tied recurrence baseline |
| `LR_0.009_regression` | 3.32 | Lower LR was a regression for 55-step budget |
| `Best_3090_12step` | 3.18 | Warmup/clip improvements stabilized the run |
| `Best_reproducible_3090` | 3.1853 | Most reproducible UT-era setting |

External context that was useful in identifying practical limits of deeper UT-style recurrence in this setting:

- [Unofficial: Interactive dashboard visualizing all 352 submissions (dentity007)](https://github.com/openai/parameter-golf/discussions/747)
- [PR #363](https://github.com/openai/parameter-golf/pull/363)

### Phase 2 — Throughput pivot

| Milestone | Val BPB | Notes |
|---|---:|---|
| `dim512_discovery` | 2.7335 | More optimizer steps in 10 minutes beat deeper slower setup |

### Phase 3 — New architecture regime

| Milestone | Val BPB | Notes |
|---|---:|---|
| `mlp2_steps2_LR012` | **1.8085** | Strong result in new single-pass/wide family |
| `mlp5_steps1_lora512_BEST` | 1.8117 | Best in journey sheet for steps=1 family |

> Important: `SmearGate_BUG` entry with 0.7690 BPB in the journey CSV is explicitly marked **invalid** due to a non-causal leak.

## Stable UT-era sweep results (10-minute window)

| Run ID | MATRIX_LR | GRAD_CLIP_NORM | TARGET_GRAD_NORM | Final BPB |
|---|---:|---:|---:|---:|
| `SWEEP_01_lr010_clip06` | 0.010 | 0.6 | 0.25 | 3.2051 |
| `SWEEP_02_lr011_clip08` | 0.011 | 0.8 | 0.25 | 3.1952 |
| `SWEEP_03_lr012_clip08` | 0.012 | 0.8 | 0.25 | 3.1918 |
| `SWEEP_04_tgn020` | 0.012 | 0.8 | 0.20 | **3.1861** |
| `SWEEP_05_tgn030` | 0.012 | 0.8 | 0.30 | 3.2002 |

Refinement follow-up:

| Run ID | Change | Final BPB | Outcome |
|---|---|---:|---|
| `REFINE_01_baseline` | baseline | 3.2075 | reference |
| `REFINE_02_warmup12` | `WARMUP_STEPS=12` | 3.3182 | worse |
| `REFINE_03_muon4` | `MUON_BACKEND_STEPS=4` | 3.2274 | worse |

## AB3 feature sweep snapshot

Source: `results/ab3_sgbo_fixed/summary.csv`

| Run | SMEARGATE | BIGRAM_HASH | ORTHO_INIT | Best BPB |
|---|---:|---:|---:|---:|
| AB3_010 | 0 | 1 | 0 | **1.8279** |
| AB3_000 | 0 | 0 | 0 | 1.8374 |
| AB3_011 | 0 | 1 | 1 | 1.8492 |
| AB3_001 | 0 | 0 | 1 | 1.8550 |
| AB3_100 | 1 | 0 | 0 | 1.8729 |
| AB3_111 | 1 | 1 | 1 | 1.8826 |
| AB3_101 | 1 | 0 | 1 | 1.8866 |
| AB3_110 | 1 | 1 | 0 | 1.8874 |

## Engineering work delivered

- Windows-safe launcher + backend patching (`train_gpt_windows.py`)
- Architecture and optimizer routing refactors (`model.py`, `train_gpt.py`)
- VRAM-safe EMA eval swap + logging instrumentation
- Dynamic LR/gradient safety controls + loss filtering
- Fused Triton MLP path (`triton_mlp.py`)
- Quantized export pipeline (`quant_utils.py`)
- Dataset diagnostics artifacts (`artifacts/region_scan_report.json`)

## Deterministic reevaluation launcher (ScaleDown)

We currently use **`ScaleDown.bat`** as the deterministic reevaluation entrypoint.

Why this launcher:
- pins deterministic data order (`DATA_DETERMINISTIC=1`, fixed `DATA_SEED`),
- fixes wallclock envelope (standard-seed reeval uses `MAX_WALLCLOCK_SECONDS=598`),
- keeps a stable architecture/training profile for apples-to-apples re-checks,
- logs all critical knobs (including `LEVEL_SIGNAL_ENABLED=0`) at launch.

Key fixed settings in `ScaleDown.bat`:
- `MODEL_DIM=1024`, `MLP_MULT=5`, `ITERATIONS=500`
- `TRAIN_BATCH_TOKENS=524288`, `GRAD_ACCUM_STEPS=16`, `TRAIN_SEQ_LEN=1024`
- `RECURRENCE_STEPS/NUM_STEPS=1`, `LORA_RANK=512`, `LORA_SCOPE=q`
- `LOSS_FILTER_ENABLED=1`, `LOSS_FILTER_WARMUP=250`, `LOSS_FILTER_Z_THRESHOLD=4.5`
- `MATRIX_LR=0.08`, `SCALAR_LR=0.015`, `EMBED_LR=0.7`, `HEAD_LR=0.008`, `TIED_EMBED_LR=0.06`
- `SMEARGATE=0`, `BIGRAM_HASH=1`, `LEVEL_SIGNAL_ENABLED=0`, `ORTHO_INIT=0`

### Historical best single-run reference (under prior 600s cap)

Reference log: **`traininglog4.txt`**

Why this is the current reproducible reference run:
- launched with the deterministic `ScaleDown.bat` profile,
- fixed seed/data order (`DATA_DETERMINISTIC=1`, `DATA_SEED=3623123517`),
- hard 10-minute envelope (`MAX_WALLCLOCK_SECONDS=600`),
- no loss-filter instability (`[filter] ... skipped=0 fallback_accepts=0`).

Key outcomes from `traininglog4.txt`:
- best checkpoint: `step:400 val_bpb:1.8184` (best_val_loss `3.0800`),
- final stride-64 eval: `step:413 val_bpb:1.8652` (`[FINAL STRIDE 64]`),
- stop reason: wallclock (`elapsed:601.1s`),
- final export source confirmed as best checkpoint (`step=400`).

This run is retained as historical context, but it is **not** used for current merge metadata because it exceeded the 600s envelope.

### Standard-seed reevaluation (42, 1337, 2024) — merge reference

Profile used: `ScaleDown.bat` with `MAX_WALLCLOCK_SECONDS=598`.

> ✅ **Repro note:** To reproduce the reported/merge BPB results in this README and `submission.json`, run via **`ScaleDown.bat`** only. Do **not** use other launchers for these reference numbers.

| Seed | Run ID | Best checkpoint BPB | Best checkpoint loss | Stop (s) | Final stride-64 BPB |
|---|---|---:|---:|---:|---:|
| 42 | `STDSEED_42_0ad47cc2` | 1.8166 | 3.0771 | 599.2 | 1.8615 |
| 1337 | `STDSEED_1337_6179f3b4` | 1.8179 | 3.0793 | 599.0 | 1.8672 |
| 2024 | `STDSEED_2024_CLEAN` | 1.8296 | 3.0990 | 598.6 | 1.8768 |

Aggregate (best-checkpoint BPB):
- mean: **1.8214**
- stddev: **0.0058**
- mean wallclock: **598.93s**

### Config provenance / TTT clarification

All values reported in this README and `submission.json` are from runs with:
- `TTT_ENABLED=0`
- `SMEARGATE_ENABLED=0`
- deterministic data seed path (`DATA_DETERMINISTIC=1`, `DATA_SEED=3623123517`)

If you see `TTT_ENABLED=1` in older/experimental launcher paths, treat that as exploratory only; it is **not** the config used for the reported BPB numbers above.

## Documentation map

- [01 Overview and Timeline](notes/01_overview_and_timeline.md)
- [02 Current Architecture](notes/02_current_architecture.md)
- [03 Training System](notes/03_training_system.md)
- [04 Experimental Results](notes/04_experimental_results.md)
- [05 Engineering Endeavors](notes/05_engineering_endeavors.md)
- [06 Validation and Next Steps](notes/06_validation_and_next_steps.md)

## Run commands

```bat
ScaleDown.bat
```
