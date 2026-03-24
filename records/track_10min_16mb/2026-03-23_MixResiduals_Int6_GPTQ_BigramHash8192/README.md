## Record: Residual Input Mixing + mixed int6 GPTQ + grouped TTT + MLP 3.5x

**val_bpb: 1.1169** (mean over 3 seeds with TTT evaluation, stride=64)

**artifact: 15.6 MB** (mean over 3 seeds)

## TLDR Changes

- Changed TTT from a flat optimizer to grouped AdamW with stronger matrix/head adaptation, while restoring standard clipping and removing the per-chunk warmup.

- Changed Architecture: Making Residual Connections Denser, Changed block input formation so each transformer block now sees a learned mix of the current stream, earlier block outputs, and the original x0, instead of only the simpler local x/x0 residual mix. This gives the model a denser residual path and lets each block reuse longer-range intermediate features directly.

## Results

| Seed | Steps | final val_loss | final val_bpb | Artifact |
|------|-------|----------|-------------------|----------|
| 1337 | 6106 | 1.8859  | 1.1169 | 15.88 MB |
| 42 | 6092 | 1.8855 | 1.1167 | 15.33 MB |
| 2024 | 6091 | 1.8864 | 1.1172 | 15.73 MB |

**val_bpb mean: 1.1169**

**val_bpb std: 0.0003**

**val_loss mean: 1.8859**


## More Details

- Architecture: 11L, 512d, Mixed residuals each layer from 2 previous layers, MHA 8/8, MLP 3.5x (1792), BigramHash 8192, XSA all layers

- Quantization: mixed int6 per-row GPTQ (clip_range=15) + Early QAT (threshold 0.5) + EMA 0.997

- TTT: Legal score-first AdamW, chunk=131072, last 2 blocks plus control params unfrozen
