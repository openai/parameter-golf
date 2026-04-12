# GDN-Hybrid + Sliding Window Attention (3-seed mean 1.02045733 BPB)

## Per-seed authoritative results

| Seed | Steps | EMA BPB | Quantized BPB | XSA BPB | Artifact bytes |
|------|------:|--------:|--------------:|--------:|---------------:|
| 42 | 1864 | 1.017723 | 1.026791 | 1.031731 | 15,313,984 |
| 1337 | 2239 | 1.007375 | 1.016586 | 1.020691 | 15,830,308 |
| 2024 | 2241 | 1.008736 | 1.017995 | 1.023138 | 15,820,201 |
| **Mean** | — | **1.011278** | **1.02045733** | **1.025187** | **15,654,831.00** |
| **Std (sample)** | — | — | **0.00553017** | — | — |

## Technique stack

1. **SP1024 tokenizer** with a GDN-hybrid backbone (`[GDN×5] → SWA → [GDN×5] → SWA_shared`).
2. **Fixed-predictor / no-TTT Track-A lane** — no eval-time or pre-quant adaptation in the scored artifact.
3. **MuonEq-R + AdamW** training mix, EMA `0.997`, late QAT threshold `0.15`.
4. **GPTQ int6 + zstd-22** packaging.
5. **Sliding-window attention side path** present in-model, but submission authority remains the pulled `quantized_bpb` values above.

