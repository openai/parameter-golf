# BW13_TapOff_Anchor_GPTQ_2k — Ablation Results

Status: completed (`seed=444`, window start `2026-04-01 21:11:22`)

Current run command (unified runner):

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-02_BW15_AllCrawler_2k/run_ablation_sequence.sh
```

Legacy per-series runner was removed in favor of the single unified crawler sequence.

External run artifacts (from pod):

- Summary TSV: `/workspace/parameter-golf-lab/crawler/2026-04-01_BW13_TapOff_Anchor_GPTQ_2k/results/summary_s444_20260401_211122.tsv`
- Control checkpoint: `/workspace/parameter-golf-lab/crawler/2026-04-01_BW13_TapOff_Anchor_GPTQ_2k/results/BW13INT-00_control_s444_20260401_211122.final_model.pt`

## Lane A — WINDOW (must retrain)

| Arm | Description | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_control | Verdict |
|-----|-------------|---------|-------------|---------|-------|------------------|---------|
| BW13INT-00 | control (tap-off Nightcrawler, naive int6) | 1.2898 | 1.27151667 | 156.63 | 10860793 | +0.000000 | baseline |
| BW13INT-01 | tap-off + anchor dim=32 | 1.2921 | 1.27427613 | 158.57 | 10691928 | +0.002759 | reject |
| BW13INT-02 | tap-off + anchor dim=64 | 1.2916 | 1.27443577 | 158.55 | 10931561 | +0.002919 | reject |

## Lane B — POST_WINDOW (sequential, no retrain)

These use `SKIP_TRAIN=1` and `INIT_MODEL_PATH=<control final_model.pt>`.

| Arm | Description | raw_bpb | int6_sw_bpb | step_ms | bytes | gptq_layers | gptq_cal_sec | delta_vs_control | Verdict |
|-----|-------------|---------|-------------|---------|-------|-------------|--------------|------------------|---------|
| BW13INT-Q0 | naive int6 on frozen control checkpoint | 1.2898 | 1.27151667 | 0.00 | 10860793 | 0 | - | +0.000000 | hold |
| BW13INT-Q1 | standard GPTQ (128x2048) | 1.2898 | 1.26958450 | 0.00 | 11507691 | 30 | 1.0 | -0.001932 | promote (full-train rerun) |
| BW13INT-Q1L | standard GPTQ-lite (64x1024) | 1.2898 | 1.26975109 | 0.00 | 11629176 | 30 | 0.6 | -0.001766 | hold (dominated by Q1) |

## Full-Run Promotion Queue (600s, 8xH100)

Promote only if gate clears noise floor:

- WINDOW arm: `delta_vs_control <= -0.0008`
- POST_WINDOW quant arm: `delta_vs_control <= -0.0008`, then rerun same quant policy with full training (`SKIP_TRAIN=0`)

| Candidate | Triggered By | Full-run status |
|-----------|--------------|-----------------|
| BW13INT-Q1 | POST_WINDOW delta=-0.001932 | pending |

## Notes

- Anchor interactions on tap-off baseline are strongly negative in this seed (`+0.0027` to `+0.0029`) and should not be promoted.
- `Q1` beats `Q0` and `Q1L` on quality; `Q1L` is faster to calibrate (0.6s vs 1.0s) but worse in both bpb and bytes.
- Immediate promotion candidate from this leg is `Q1` as a full training run (`SKIP_TRAIN=0`).
