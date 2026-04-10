# Helix — Ablation Results

Status: pending

## Gate (1k steps, 1-GPU, seed=444)

| Arm | Config | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_ctrl |
|-----|--------|---------|-------------|---------|-------|---------------|
| Control | HELIX=0 (Ouroboros base) | | | | | +0.000000 |
| Stride=3 | HELIX=1 HELIX_STRIDE=3 (3 passes) | | | | | |
| Stride=1 | HELIX=1 HELIX_STRIDE=1 (9 passes) | | | | | |

## Notes
- Control is Ouroboros config without helix (sequential encoder→crawler→decoder)
- Stride=3: crawler fires at flat layers 3, 6, 9 (safe compute budget)
- Stride=1: crawler fires at every flat layer (ambitious, ~9 crawler passes)
- Watch step_ms — more crawler passes = slower steps = fewer total steps in 600s
- Zero-init warm start means helix starts as no-op and learns to inject
