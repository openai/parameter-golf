# BW14_BigSwing_2k — Ablation Results

Status: in progress (4/5 WINDOW arms logged)

Current run command (unified runner):

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-02_BW15_AllCrawler_2k/run_ablation_sequence.sh
```

Legacy per-series runner was removed in favor of the single unified crawler sequence.

Summary file is emitted to:

- `crawler/2026-04-01_BW14_BigSwing_2k/results/summary_s<seed>_<timestamp>.tsv`

## WINDOW Arms (must retrain)

| Arm | Description | model_params | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_control | Verdict |
|-----|-------------|--------------|---------|-------------|---------|-------|------------------|---------|
| BW14BS-00 | control (tap-off Nightcrawler, naive int6) | 16,823,860 | 1.2936 | 1.27383046 | 94.26 | 11,074,768 (int6+zlib) | +0.000000 | baseline |
| BW14BS-01 | depth phase shift: NUM_FLAT_LAYERS=6 | 19,185,724 | 1.2867 | 1.26683545 | 104.50 | 12,516,632 (int6+zlib) | -0.006995 | promote (big-swing hit) |
| BW14BS-02 | crawler choke flat-128 | 16,627,252 | 1.3051 | 1.28551734 | 94.63 | 11,401,230 (int6+zlib) | +0.011687 | reject |
| BW14BS-03 | crawler choke flat-512 | 20,756,020 | 1.2934 | 1.27377940 | 97.14 | 13,529,127 (int6+zlib) | -0.000051 | no signal |
| BW14BS-04 | crawler choke residual-128 | pending | pending | pending | pending | pending | pending | pending |

## Full-Run Promotion Queue (600s, 8xH100)

- Big-swing promote: `delta_vs_control <= -0.0060`
- Secondary promote: `-0.0060 < delta_vs_control <= -0.0030`
- Guardrails: artifact <= 16MB and no catastrophic speed regression.

| Candidate | Triggered By | Full-run status |
|-----------|--------------|-----------------|
| BW14BS-01 | delta_vs_control = -0.006995 (meets big-swing promote cutoff <= -0.0060) | pending |

## Notes

- This leg intentionally prefers high-amplitude architecture levers over quant micro-optimizations.
- Positive deltas are acceptable outcomes if they eliminate dead branches quickly.
- Source run (pasted logs): `/workspace/parameter-golf/crawler/2026-04-01_BW14_BigSwing_2k/results/*_20260402_011028.log`
- Environment note: `zstandard` missing on this run, so artifact reporting used `int6+zlib` instead of `int6+zstd`.
