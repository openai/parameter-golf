# BW14_BigSwing_2k — Ablation Results

Status: pending run

Run command:

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-01_BW14_BigSwing_2k/run_ablation_sequence.sh
```

Summary file is emitted to:

- `crawler/2026-04-01_BW14_BigSwing_2k/results/summary_s<seed>_<timestamp>.tsv`

## WINDOW Arms (must retrain)

| Arm | Description | model_params | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_control | Verdict |
|-----|-------------|--------------|---------|-------------|---------|-------|------------------|---------|
| BW14BS-00 | control (tap-off Nightcrawler, naive int6) | pending | pending | pending | pending | pending | — | pending |
| BW14BS-01 | depth phase shift: NUM_FLAT_LAYERS=6 | pending | pending | pending | pending | pending | pending | pending |
| BW14BS-02 | crawler choke flat-128 | pending | pending | pending | pending | pending | pending | pending |
| BW14BS-03 | crawler choke flat-512 | pending | pending | pending | pending | pending | pending | pending |
| BW14BS-04 | crawler choke residual-128 | pending | pending | pending | pending | pending | pending | pending |

## Full-Run Promotion Queue (600s, 8xH100)

- Big-swing promote: `delta_vs_control <= -0.0060`
- Secondary promote: `-0.0060 < delta_vs_control <= -0.0030`
- Guardrails: artifact <= 16MB and no catastrophic speed regression.

| Candidate | Triggered By | Full-run status |
|-----------|--------------|-----------------|
| pending | pending | pending |

## Notes

- This leg intentionally prefers high-amplitude architecture levers over quant micro-optimizations.
- Positive deltas are acceptable outcomes if they eliminate dead branches quickly.
