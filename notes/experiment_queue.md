# Autoresearch Experiment Queue

## Current Best
- Config: heavycycle3 + wd=0.01 + smear(-3.0) + MoD(keep=0.75, core=1)
- val_bpb: 2.02095 (seed 2026) / 2.02522 (seed 1337)
- Compressed: ~3.66 MB

## Baseline Command
```bash
cd /Users/soumil/Desktop/parameter-golf && \
DEPTH_SHARE_MODE=cycle DEPTH_UNIQUE_LAYERS=3 DEPTH_SHARE_HEAVY_ONLY=1 \
MUON_WEIGHT_DECAY=0.01 SMEARGATE=1 SMEARGATE_INIT=-3.0 \
MOD_KEEP=0.75 MOD_CORE=1 \
ATTNRES_MODE=none LATENT_MEM_MODE=none \
MAX_WALLCLOCK_SECONDS=180 VAL_LOSS_EVERY=0 SEED=1337 \
RUN_ID=baseline_check \
python train_gpt_mlx.py
```

## Running
| ID | Description | Seed | Status |
|----|-------------|------|--------|
| auto_01 | WARMDOWN_ITERS=300 | 1337 | running |
| auto_02 | WARMDOWN_ITERS=200 | 1337 | running |

## Pending (Wave 2+ use best WARMDOWN from Wave 1)
| Wave | ID | Priority | Type | Description | Changes |
|------|----|----------|------|-------------|---------|
| 2 | auto_03 | P1 | hp | SmearGate -4.0 | SMEARGATE_INIT=-4.0 |
| 2 | auto_04 | P1 | hp | 12 layers / 3 cores | NUM_LAYERS=12 |
| 3 | auto_05 | P1 | hp | MOD_KEEP 0.625 | MOD_KEEP=0.625 |
| 3 | auto_06 | P1 | hp | Muon WD 0.02 | MUON_WEIGHT_DECAY=0.02 |
| 4 | auto_07 | P2 | hp | MLP LoRA rank 8 | MLP_LORA_RANK=8 |
| 4 | auto_08 | P2 | hp | Grad clip 1.0 | GRAD_CLIP_NORM=1.0 |
| 5 | auto_09 | P2 | hp | SmearGate -5.0 | SMEARGATE_INIT=-5.0 |
| 5 | auto_10 | P2 | hp | 2 unique cores | DEPTH_UNIQUE_LAYERS=2 |
| 6 | auto_11 | P2 | hp | Seq len curriculum | SEQ_LEN_SCHEDULE=linear SEQ_LEN_MIN=256 SEQ_LEN_RAMP_STEPS=5000 |
| 6 | auto_12 | P2 | hp | Logit softcap 15 | LOGIT_SOFTCAP=15.0 |
| 7 | auto_13 | P3 | code | NorMuon optimizer | code change |
| 7 | auto_14 | P3 | code | Progressive residual warmup | code change |
| 8 | auto_15 | P3 | hp | 15 layers | NUM_LAYERS=15 |
| - | combo | P4 | combo | Best winners combined | TBD |

## Completed
| ID | Description | val_bpb | Compressed | vs Best | Seed | Log |
|----|-------------|---------|------------|---------|------|-----|
