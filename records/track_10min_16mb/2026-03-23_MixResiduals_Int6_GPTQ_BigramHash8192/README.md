## Record: Residual Input Mixing + mixed int6 GPTQ + grouped TTT + MLP 3.5x

**val_bpb: 1.1172** (mean over 3 seeds with TTT evaluation, stride=64)

**artifact: 15.5 MB** (mean over 3 seeds)

## TLDR Changes

- in my previous submission (PR #615) there was a bug, which is now fixed: GPTQ calibration time is now counted as a part of training time, meaning that the 600s constraint does affects it. In order to incorporate that, the training loop's wall clock time was lowered a few seconds so that the total time would stay under 600s.

- Changed TTT from a flat optimizer to grouped AdamW (separating params in two groups: matrices and control weights) with stronger matrix/head adaptation, while restoring standard clipping and removing the per-chunk warmup.

- Changed Architecture: Making Residual Connections Denser, Changed block input formation so each transformer block now sees a learned mix of the current stream, earlier block outputs, and the original x0, instead of only the simpler local x/x0 residual mix. This gives the model a denser residual path and lets each block reuse longer-range intermediate features directly.

## Results

| Seed | Steps | final val_loss | final val_bpb | Artifact |
|------|-------|----------|-------------------|----------|
| 1337 | 6034 | 1.8869 | 1.1176 | 15.3 MB |
| 42 | 6031 | 1.8856 | 1.1168 | 15.9 MB |
| 2025 | 6033 | 1.8863 | 1.1172 | 15.4 MB |

**val_bpb mean: 1.1172**

**val_bpb std: 0.0003**

**val_loss mean: 1.8863**


## More Details

- Architecture: 11L, 512d, Mixed residuals each layer from 2 previous layers, MHA 8/8, MLP 3.5x (1792), BigramHash 8192, XSA all layers

- Quantization: mixed int6 per-row GPTQ (clip_range=15) + Early QAT (threshold 0.5) + EMA 0.997

- TTT: Legal score-first AdamW, chunk=131072, last 2 blocks plus control params unfrozen
