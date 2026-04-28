# BW15_AllCrawler_2k — Ablation Log

Status: ready to run on 4x pod

Run command:

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-02_BW15_AllCrawler_2k/run_ablation_sequence.sh
```

Summary output:

- `crawler/2026-04-02_BW15_AllCrawler_2k/results/summary_s<seed>_<timestamp>.tsv`

## Ordering

1. `BIG_SWING` phase first.
2. `SMALL` phase second.

## Arm Inventory

| Phase | Lane | Arm | Description | Control Group | Notes |
|------|------|-----|-------------|---------------|-------|
| BIG_SWING | WINDOW | BW14BS-00 | control (tap-off Nightcrawler, naive int6) | tapoff | retrain |
| BIG_SWING | WINDOW | BW14BS-01 | depth phase shift: NUM_FLAT_LAYERS=6 | tapoff | retrain |
| BIG_SWING | WINDOW | BW14BS-02 | crawler choke flat-128 | tapoff | retrain |
| BIG_SWING | WINDOW | BW14BS-03 | crawler choke flat-512 | tapoff | retrain |
| BIG_SWING | WINDOW | BW14BS-04 | crawler choke residual-128 | tapoff | retrain |
| SMALL | WINDOW | BW13INT-00 | control (tap-off Nightcrawler, naive int6) | tapoff | alias reuse of BW14BS-00 |
| SMALL | WINDOW | BW13INT-01 | tap-off + anchor dim=32 | tapoff | retrain |
| SMALL | WINDOW | BW13INT-02 | tap-off + anchor dim=64 | tapoff | retrain |
| SMALL | POST_WINDOW | BW13INT-Q0 | naive int6 on frozen tap-off control | tapoff | sequential |
| SMALL | POST_WINDOW | BW13INT-Q1 | standard GPTQ (128x2048) on frozen tap-off control | tapoff | sequential |
| SMALL | POST_WINDOW | BW13INT-Q1L | standard GPTQ-lite (64x1024) on frozen tap-off control | tapoff | sequential |
| SMALL | WINDOW | BW12INT-00 | control (Nightcrawler 5F + TAP shared, naive int6) | tapshared | retrain |
| SMALL | WINDOW | BW12INT-01 | tap off (isolate 5F depth without tap) | tapshared | retrain |
| SMALL | WINDOW | BW12INT-02 | anchor dim=32 on Nightcrawler stack | tapshared | retrain |
| SMALL | POST_WINDOW | BW12INT-Q0 | naive int6 on frozen tap-shared control | tapshared | sequential |
| SMALL | POST_WINDOW | BW12INT-Q1 | standard GPTQ on frozen tap-shared control | tapshared | sequential |
| SMALL | POST_WINDOW | BW12INT-Q2 | loop-aware GPTQ on frozen tap-shared control | tapshared | sequential |

## Promotion Guidance

- Big-swing promote: `delta_vs_control <= -0.0060`
- Small-sweep promote: `delta_vs_control <= -0.0008`

