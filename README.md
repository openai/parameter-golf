# parameter_golf_bj_run

This repository contains my local reproduction of the OpenAI Parameter Golf baseline.

## 🚀 Baseline Result (Local Run)

- Environment: single GPU (local)
- Dataset: fineweb10B_sp1024 (train_shards=1)
- Sequence length: 1024
- Grad accumulation: 8

### Best Result

- **val_bpb = 1.3529 @ step 4200**

### Training Summary

- step 0: val_bpb = 4.1077
- step 1000: val_bpb = 1.3944
- step 2000: val_bpb = 1.3651
- step 2800: val_bpb = 1.3548
- step 4200: val_bpb = 1.3529 (best)
- step 5000: val_bpb = 1.3576

## 📊 Observations

- Rapid improvement in early training
- Convergence around ~1.35 bpb
- No further improvement after ~4200 steps
- Additional training leads to fluctuation, not improvement

## 📁 Files

- Log file:
  - `local_logs/baseline_local_smoke_2026_03_21_v01.log`

- Experiment notes:
  - `notes/baseline_run_2026-03-21.md`

## 🎯 Status

- Baseline successfully reproduced on local hardware
- This result is used as the reference point for further experiments

## 🔜 Next Steps

- Modify training parameters (e.g. learning rate)
- Add evaluation improvements (e.g. sliding window)
- Compare against baseline (1.3529)

---

> This repository is based on the OpenAI Parameter Golf baseline, with added local experiment results.
