# EPSILON: 11L Maximum Expressivity Profile

## Status
- Attention path: flash-only (sparse routing removed) but scaled up.
- Focus: Maximizing theoretical BPB by fully utilizing the 8xH100 10-minute constraint. Thermodynamic LR and Riemannian Sphere optimizer are retained to ensure fast convergence.
- Expected score: < 1.1147 val_bpb

## Current Default Hyperparameters
```text
NUM_LAYERS=11
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=3.0

XSA_LAST_N=11
VE_ENABLED=1
BIGRAM_VOCAB_SIZE=2048
BIGRAM_DIM=128

SWA_ENABLED=1
SWA_EVERY=40
MUON_WD=0.06
ADAM_WD=0.06
TRAIN_SEQ_LEN=2048
```

## Benchmark Target
The benchmark target is the current README top-1 run:
`records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`

This folder includes `sim_checks.py`, which compares this run directly against that top-1 run on the same hardware.

## Run Benchmark
```powershell
python sim_checks.py
```

Optional knobs:
```powershell
$env:SIM_SEQ_LEN="512"
$env:SIM_BATCH="2"
$env:SIM_ITERS="8"
$env:SIM_WARMUP="3"
$env:SIM_WINNER_SPEED_MIN="1.0"
python sim_checks.py
```

Passing condition:
- `speed_ratio_vs_top1 >= SIM_WINNER_SPEED_MIN`
