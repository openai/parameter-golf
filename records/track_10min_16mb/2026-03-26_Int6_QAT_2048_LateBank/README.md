# Int6 QAT 2048 + Late Bank QAT + MLP_MULT=2.75

**val_bpb: TBD** (3-seed mean, post int6+lzma, sliding window + TTT)

## Summary

**Unique contribution:** Late QAT applied to parameter bank weights (Q, K, V, O, MLP up/down)
during warmdown. #1 leaderboard entry only applies late QAT to CastedLinear layers (bigram,
VE, lm_head) — the minor ~5% of weights. This experiment extends late QAT to the core 95%
of weights via `_fq_bank()` STE fake-quant, activated when lr_scale < BANK_QAT_THRESHOLD=0.15.

Key settings:
- `TRAIN_SEQ_LEN=2048`, `EVAL_SEQ_LEN=2048`
- `MLP_MULT=2.75` (down from 3.0) — reduces artifact to ~15.85MB, under 16MB limit
- `QAT_ENABLED=1` — CastedLinear fake-quant from step 1 (minor weights)
- `BANK_QAT_THRESHOLD=0.15` — bank QAT activates when lr_scale < 0.15 (warmdown)
- Export: GPTQ-lite int6 + lzma

**Why late and not from step 1:** Exp6 showed that bank QAT from step 1 disrupts training
(post-roundtrip BPB degraded from 1.1407 to 1.1489). Waiting until the model is converged
before adding int6 noise to banks lets the model first learn good representations, then
adapt cleanly to quantization in the final ~15% of training.

**Expected BPB:** ~1.120–1.125 post-TTT if late bank QAT reduces roundtrip degradation
vs both #1 (no bank QAT) and Exp6 (bank QAT from step 1).

## Results

| Seed | Steps | Pre-TTT bpb | Post-TTT bpb | Artifact (bytes) |
|------|-------|-------------|--------------|-----------------|
| 1337 | — | — | — | — |
| 42   | — | — | — | — |
| 2025 | — | — | — | — |
| **Mean** | | | | |
