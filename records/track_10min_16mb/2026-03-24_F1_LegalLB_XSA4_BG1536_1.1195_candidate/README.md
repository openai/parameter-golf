# Candidate: F1 Legal-LB Profile (XSA4 + Bigram1536) — 1.1195

This entry logs the 8xH100 run from `concepts/f1/run_legal_lb.sh` on **March 24, 2026**.

## Run Provenance

- Runner: `concepts/f1/run_legal_lb.sh`
- Source script: `concepts/f1/train_gpt.py` (copied here as `train_gpt.py`)
- Pod hardware: 8x NVIDIA H100 80GB HBM3
- Seed: `1337`
- Pod run id: `f1_legal_lb_s1337_20260324_215935`
- Pod log path printed by run: `logs/f1_legal_lb_s1337_20260324_215935.txt`

## Config Deltas vs F1 Baseline

Only the legal leaderboard profile knobs were applied for this run:

- `MLP_ACT=leaky_relu_sq`, `MLP_LEAKY_SLOPE=0.5`
- `XSA_LAST_N=4`
- `BIGRAM_VOCAB_SIZE=1536`
- `TTT_FREEZE_BLOCKS=0`, `TTT_GRAD_CLIP=0.8`
- `F1_CORR_RANK=0`, `DISTILL_ENABLED=0`

## Key Metrics (from console log)

- `model_params: 26928220`
- `step_avg` near stop: `86.72ms`
- Train wallclock stop: `600021ms` at step `6919`
- `DIAGNOSTIC post_ema val_bpb: 1.1379`
- `final_int6_roundtrip_exact val_bpb: 1.14332344`
- `final_int6_sliding_window_exact val_bpb: 1.11959640`
- `legal_ttt_exact val_bpb: 1.11951975`
- Serialized model int6+zstd: `15809827 bytes`
- Total submission size int6+zstd: `15901632 bytes` (under 16MB)
- TTT eval time: `223102ms`

## Rule Checklist Review

- Under 16MB artifact: **PASS** (`15,901,632 bytes`).
- 8xH100 environment: **PASS** (confirmed in run output).
- 0.005-nat improvement bar for *new SOTA*: **NOT YET CLEARLY PASS**.
  - Against PR #587 figure (`1.1203/1.1204`), this is ~`0.0008` better, not `0.005`.
- Multi-run significance requirement (`p < 0.01`): **NOT MET** (single seed logged here).
- Under-10-minute evaluation requirement: **RISK / NEEDS CLARIFICATION**.
  - Training capped at ~600s, but quant+eval+TTT adds substantial extra runtime.

## Status

This is logged as a **candidate run** (strong result, not yet a safe official SOTA claim under strict submission criteria).
