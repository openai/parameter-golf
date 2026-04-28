# BW12_Interaction_2k — Ablation Results

Status: completed (`seed=444`, window start `2026-04-01 20:26:18`)

Current run command (unified runner):

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-02_BW15_AllCrawler_2k/run_ablation_sequence.sh
```

Legacy per-series runner was removed in favor of the single unified crawler sequence.

External run artifacts (from pod):

- Summary TSV: `/workspace/parameter-golf-lab/crawler/2026-04-01_BW12_Interaction_2k/results/summary_s444_20260401_202618.tsv`
- Control checkpoint: `/workspace/parameter-golf-lab/crawler/2026-04-01_BW12_Interaction_2k/results/BW12INT-00_control_s444_20260401_202618.final_model.pt`

## Lane A — WINDOW (must retrain)

| Arm | Description | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_control | Verdict |
|-----|-------------|---------|-------------|---------|-------|------------------|---------|
| BW12INT-00 | control (Nightcrawler 5F+TAP shared, naive int6) | 1.2925 | 1.27438319 | 158.49 | 10867859 | +0.000000 | baseline |
| BW12INT-01 | tap off (isolate depth-only behavior) | 1.2904 | 1.27238951 | 156.96 | 10808331 | -0.001994 | promote |
| BW12INT-02 | anchor dim=32 on Nightcrawler stack | 1.2909 | 1.27336822 | 160.23 | 10702537 | -0.001015 | promote |

## Lane B — POST_WINDOW (sequential, no retrain)

These use `SKIP_TRAIN=1` and `INIT_MODEL_PATH=<control final_model.pt>`.

| Arm | Description | raw_bpb | int6_sw_bpb | step_ms | bytes | gptq_layers | delta_vs_control | Verdict |
|-----|-------------|---------|-------------|---------|-------|-------------|------------------|---------|
| BW12INT-Q0 | naive int6 on frozen control checkpoint | 1.2925 | 1.27438319 | 0.00 | 10867859 | 0 | +0.000000 | hold |
| BW12INT-Q1 | standard GPTQ on frozen control checkpoint | 1.2925 | 1.27233966 | 0.00 | 11634167 | 30 | -0.002044 | promote (full-train rerun) |
| BW12INT-Q2 | loop-aware GPTQ on frozen control checkpoint | 1.2925 | 1.27233966 | 0.00 | 11634167 | 30 | -0.002044 | tie with Q1 |

## Full-Run Promotion Queue (600s, 8xH100)

Promotion gate:

- WINDOW arm: `delta_vs_control <= -0.0008`
- POST_WINDOW quant arm: `delta_vs_control <= -0.0008`, then rerun same quant policy with full training (`SKIP_TRAIN=0`)

| Candidate | Triggered By | Full-run status |
|-----------|--------------|-----------------|
| BW12INT-01 | WINDOW delta=-0.001994 | pending |
| BW12INT-02 | WINDOW delta=-0.001015 | pending |
| BW12INT-Q1 | POST_WINDOW delta=-0.002044 | pending |
| BW12INT-Q2 | POST_WINDOW delta=-0.002044 (tie with Q1) | optional/pending |

## Notes

- Both post-window GPTQ variants (Q1/Q2) matched exactly on final metric and size in this 2k signal run.
- Q1 is operationally simpler and should be prioritized first for full-train confirmation unless Q2 is explicitly being stress-tested.
- All arms above are 2k-step gates and require full 600s confirmation before stack promotion.
