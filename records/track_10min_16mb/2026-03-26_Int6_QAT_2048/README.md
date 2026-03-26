# Int6 QAT from Step 1 — 2048 Context

**val_bpb: TBD** (3-seed mean, post int6+lzma, sliding window + TTT)

## Summary

Same as `LongContext4096_Int6_QAT` but with context length reduced to 2048.

**Unique contribution:** QAT (int6 fake-quant) enabled from step 1 throughout all training,
vs. #1 leaderboard entry which uses late QAT only during warmdown.

Key settings:
- `TRAIN_SEQ_LEN=2048`, `EVAL_SEQ_LEN=2048`
- `QAT_ENABLED=1` (default, int6 STE fake-quant from step 1)
- `late_qat_threshold=0.0` (QAT is always on, no late trigger)
- Export: GPTQ-lite int6 + lzma

**Why 2048:** 4096 context gave ~98ms/step (~6000 steps in 10min) and 16.7MB artifact.
2048 gives ~83ms/step (~7200 steps) and fits under 16MB — matching #1's throughput
while keeping our early-QAT differentiator.

**Expected BPB:** ~1.118–1.122 post-TTT if early QAT matches or beats #1's 1.1194.

## Results

| Seed | Steps | Pre-TTT bpb | Post-TTT bpb | Artifact (bytes) |
|------|-------|-------------|--------------|-----------------|
| 1337 | — | — | — | — |
| 42   | — | — | — | — |
| 2025 | — | — | — | — |
| **Mean** | | | | |
