# Late Bank QAT + 2048 Train / 4096 Eval

**val_bpb: TBD** (3-seed mean, post int6+lzma, sliding window + TTT)

## Summary

Train at 2048 context for fast step times (~83ms, ~7200 steps in 10min), but evaluate
at 4096 context for better BPB. Full MLP (3.0x) with bank QAT keeping artifact under 16MB.

**Unique contributions:**
1. **Late Bank QAT**: int6 STE fake-quant applied to all parameter bank weights
   (Q, K, V, O, MLP up/down) during warmdown via `_fq_bank()`. No other submission
   applies QAT to bank weights. Activates at `BANK_QAT_THRESHOLD=0.15`.
2. **Split context**: Train at 2048 (speed), eval at 4096 (quality).

Key settings:
- `TRAIN_SEQ_LEN=2048`, `EVAL_SEQ_LEN=4096`
- `MLP_MULT=3.0` (full capacity)
- `QAT_ENABLED=0`, `LATE_QAT_THRESHOLD=0.15` (standard CastedLinear late QAT)
- `BANK_QAT_THRESHOLD=0.15` (bank QAT during warmdown)
- `WARMDOWN_ITERS=3500`
- Export: GPTQ-lite int6 + lzma

## Results

| Seed | Steps | Pre-TTT bpb | Post-TTT bpb | Artifact (bytes) |
|------|-------|-------------|--------------|-----------------|
| 1337 | — | — | — | — |
| 42   | — | — | — | — |
| 2025 | — | — | — | — |
| **Mean** | | | | |
