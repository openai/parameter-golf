# BW17_DGXSpark_Cadence_Longform — Ablation Log

Status: ready

Single-run command:

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-02_BW17_DGXSpark_Cadence_Longform/run_ablation_sequence.sh
```

Rapid-only mode:

```bash
SEED=444 NPROC_PER_NODE=4 RUN_LONGFORM=0 bash crawler/2026-04-02_BW17_DGXSpark_Cadence_Longform/run_ablation_sequence.sh
```

Summary output:
- `crawler/2026-04-02_BW17_DGXSpark_Cadence_Longform/results/summary_s<seed>_<timestamp>.tsv`

Notes:
- RAPID stage uses smaller tokens by default (`TRAIN_BATCH_TOKENS=393216`) for quick local ranking.
- LONGFORM stage replays only control + top rapid candidates at 600s.
- POST_WINDOW quant runs on the best LONGFORM checkpoint.

## Table Template

| phase | lane | arm | source_arm | desc | model_params | raw_bpb | int6_sw_bpb | step_ms | bytes | gptq_layers | gptq_cal_sec | delta_vs_control | verdict |
|------|------|-----|------------|------|--------------|---------|-------------|---------|-------|-------------|--------------|------------------|---------|
| RAPID | WINDOW | BW17DGX-00 | BW17DGX-00 | control cadence |  |  |  |  |  | 0 | - | +0.000000 | baseline |
| RAPID | WINDOW | BW17DGX-01..07 | BW17DGX-* | cadence interactions |  |  |  |  |  | 0 | - |  |  |
| LONGFORM | WINDOW | BW17L-00 | BW17DGX-00 | long control replay |  |  |  |  |  | 0 | - | +0.000000 | baseline |
| LONGFORM | WINDOW | BW17L-01.. | top rapid sources | long candidate replays |  |  |  |  |  | 0 | - |  |  |
| LONGFORM | POST_WINDOW | BW17Q-00 | best long source | naive int6 |  |  |  | 0.00 |  | 0 | - | +0.000000 | baseline |
| LONGFORM | POST_WINDOW | BW17Q-01 | best long source | GPTQ standard |  |  |  | 0.00 |  |  |  |  |  |
| LONGFORM | POST_WINDOW | BW17Q-01L | best long source | GPTQ-lite |  |  |  | 0.00 |  |  |  |  |  |
| LONGFORM | POST_WINDOW | BW17Q-02 (optional) | best long source | loop-aware GPTQ |  |  |  | 0.00 |  |  |  |  |  |
