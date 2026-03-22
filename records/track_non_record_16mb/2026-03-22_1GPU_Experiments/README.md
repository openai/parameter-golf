# 1GPU Experiments — Feature Isolation

Single H100 development runs to cleanly test features that crashed on 8×H100 DDP.

## Why 1 GPU

Our 8×H100 runs hit DDP incompatibilities with BACKOUT, layer drop, and head drop (torch.compile + DDPOptimizer "higher order op" errors on Python 3.11 pods). Single GPU avoids all DDP issues and costs $2-3/hr vs $21/hr.

## Experiment plan

Test one feature at a time against the proven 1.1375 baseline. Compare val_bpb at matching step counts.

| Experiment | What it tests | Expected gain | Time |
|-----------|--------------|---------------|------|
| `baseline` | Reference — proven 1.1375 config | — | 30 min |
| `backout` | BACKOUT=1 (learned residual subtraction) | -0.007 BPB | 30 min |
| `wd20k` | WARMDOWN_ITERS=20000 | -0.005-0.009 BPB | 30 min |
| `swa` | Tight SWA instead of EMA | -0.002-0.005 BPB | 30 min |
| `backout_wd20k` | BACKOUT + WD20K together | -0.010-0.015 BPP | 30 min |
| `full_stack` | All proven improvements | -0.015-0.020 BPB | 2 hours |
| `moonshot` | + Reptile TTT + Shared VE + GPTQ-lite | -0.020-0.030 BPB | 2 hours |

## Usage

```bash
cd /workspace/parameter-golf
bash records/track_non_record_16mb/2026-03-22_1GPU_Experiments/run_experiment.sh baseline
bash records/track_non_record_16mb/2026-03-22_1GPU_Experiments/run_experiment.sh backout
bash records/track_non_record_16mb/2026-03-22_1GPU_Experiments/run_experiment.sh moonshot
```

## Key questions to answer

1. Does BACKOUT actually improve val_bpb? (Never tested due to DDP crashes)
2. Does WD=20000 improve the FINAL roundtrip score? (Pod crashed before completion)
3. Does tight SWA outperform EMA? (Saves 3ms/step, potentially better quant roundtrip)
4. Does the artifact fit under 16MB with smooth weights + pruning? (Untested)
5. Do Reptile TTT and Shared VE improve over plain TTT? (Novel, untested)

## Results

| Experiment | val_bpb | Steps | Artifact | Notes |
|-----------|---------|-------|----------|-------|
| baseline | TBD | | | Reference — proven 1.133 config |
| two_phase_ttt | TBD | | | **PRIORITY** — the 1.12x technique |
| reptile_ttt | TBD | | | Reptile + two-phase TTT |
| ve | TBD | | | Shared Value Embedding |
| backout | TBD | | | BACKOUT=1 isolation |
| wd20k | TBD | | | WD=20000 isolation |
| swa | TBD | | | Tight SWA vs EMA |
| backout_wd20k | TBD | | | Stack test |
| full_stack | TBD | | | All proven improvements |
| moonshot | TBD | | | Everything stacked |

Winners go into the 8×H100 10-min competition run.

## What translates from 1 GPU → 8 GPU

**Translates directly (same result on both):**
- Whether a feature improves val_bpb (relative gain is the same)
- Quantization roundtrip quality (same weight distribution → same compression)
- Artifact size (same model, same quantization, same zstd)
- Post-training techniques (TTT, Reptile, GPTQ-lite — identical on 1 or 8 GPUs)

**Doesn't translate exactly:**
- Absolute val_bpb numbers — 1 GPU sees fewer tokens in 30 min. Scores will be worse. Compare RELATIVE to baseline, not absolute.
- Step timing — 1 GPU step_avg tells us nothing about 8 GPU throughput
- Gradient noise at small batch — mitigated by using same effective batch (524K tokens with grad_accum=8)

**How to compare:**
Use val_bpb at the SAME STEP COUNT across experiments, not the same wallclock. If BACKOUT improves val_bpb by 0.007 at step 3000 on 1 GPU, it'll improve by ~0.007 at step 3000 on 8 GPUs. The relative gains transfer.

**Workflow:**
1. Run 1 GPU experiments ($1.50 each, 30 min)
2. Identify which features improve relative to baseline
3. Stack only the winners into an 8×H100 competition run
4. Submit the 8×H100 result to PR #212
