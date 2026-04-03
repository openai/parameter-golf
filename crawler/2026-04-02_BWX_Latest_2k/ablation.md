# BWX_Latest_2k — Ablation Log

Status: queued/running

Primary series: this is the main BWX contender selection sequence.

Run command:

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-02_BWX_Latest_2k/run_ablation_sequence.sh
```

Optional loop-aware post-window arm:

```bash
SEED=444 NPROC_PER_NODE=4 RUN_LOOP_AWARE_GPTQ=1 bash crawler/2026-04-02_BWX_Latest_2k/run_ablation_sequence.sh
```

Summary output:
- `crawler/2026-04-02_BWX_Latest_2k/results/summary_s<seed>_<timestamp>.tsv`
- Logs: `crawler/2026-04-02_BWX_Latest_2k/results/BWXLT-*_s<seed>_<timestamp>.log`
- WINDOW checkpoints: `crawler/2026-04-02_BWX_Latest_2k/results/BWXLT-*_s<seed>_<timestamp>.final_model.pt`

Series arms:
- WINDOW: `BWXLT-00` (8F control contender), `BWXLT-06` (6F big swing retest), `BWXLT-07`, `BWXLT-09`.
- POST_WINDOW (on best WINDOW checkpoint): `BWXLT-Q0`, `BWXLT-Q1`, `BWXLT-Q1L` (and optional `BWXLT-Q2` when enabled).

## Table Template

| lane | arm | description | num_flat_layers | source_ckpt | model_params | raw_bpb | int6_sw_bpb | step_ms | bytes | gptq_layers | gptq_cal_sec | delta_vs_control | verdict |
|------|-----|-------------|-----------------|-------------|--------------|---------|-------------|---------|-------|-------------|--------------|------------------|---------|
| WINDOW | BWXLT-00 | control contender (tap-off, no anchor, 8F, naive int6) | 8 | - |  |  |  |  |  | 0 | - | +0.000000 | baseline |
| WINDOW | BWXLT-06 | big swing retest (6F) | 6 | - |  |  |  |  |  | 0 | - |  |  |
| WINDOW | BWXLT-07 | depth sanity below contender | 7 | - |  |  |  |  |  | 0 | - |  |  |
| WINDOW | BWXLT-09 | depth sanity above contender (size-risk) | 9 | - |  |  |  |  |  | 0 | - |  |  |
| POST_WINDOW | BWXLT-Q0 | naive int6 on best WINDOW checkpoint | (best) | `<best_window_ckpt>` |  |  |  | 0.00 |  | 0 | - |  |  |
| POST_WINDOW | BWXLT-Q1 | standard GPTQ (128x2048) on best WINDOW checkpoint | (best) | `<best_window_ckpt>` |  |  |  | 0.00 |  |  |  |  |  |
| POST_WINDOW | BWXLT-Q1L | GPTQ-lite (64x1024) on best WINDOW checkpoint | (best) | `<best_window_ckpt>` |  |  |  | 0.00 |  |  |  |  |  |
| POST_WINDOW | BWXLT-Q2 (optional) | loop-aware GPTQ on best WINDOW checkpoint | (best) | `<best_window_ckpt>` |  |  |  | 0.00 |  |  |  |  |  |
