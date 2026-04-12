# GDN-Hybrid + Sliding Window Attention + compressed-code warmdown1000 (cold-cache 3-seed mean 1.01671233 BPB)

- **3-seed mean:** **1.01671233 BPB**
- **3-seed std:** **0.00134386 BPB**
- **Best seed:** **1.015700 BPB** (`seed 1337`)
- **Worst seed:** `1.018237 BPB` (`seed 2024`)
- **Artifact size range:** `15,713,422` to `15,903,365` bytes

## Per-seed authoritative results

| Seed | Steps | EMA BPB | Quantized BPB | XSA BPB | Artifact bytes |
|------|------:|--------:|--------------:|--------:|---------------:|
| 42 | 2227 | 1.007164 | 1.016200 | 1.021202 | 15,733,879 |
| 1337 | 2242 | 1.007164 | 1.015700 | 1.020105 | 15,903,365 |
| 2024 | 2227 | 1.009032 | 1.018237 | 1.024111 | 15,713,422 |
| **Mean** | — | **1.007787** | **1.01671233** | **1.021806** | **15,783,555.33** |
| **Std (sample)** | — | — | **0.00134386** | — | — |

## Technique stack

1. **SP1024 tokenizer** with a GDN-hybrid backbone (`[GDN×5] → SWA → [GDN×5] → SWA_shared`).
2. **Fixed-predictor / no-TTT Track-A lane** — no eval-time or pre-quant adaptation in the scored artifact.
3. **MuonEq-R + AdamW** training mix, EMA `0.997`, late QAT threshold `0.15`, and **warmdown=1000**.
4. **GPTQ int6 + zstd-22** packaging.
5. **Compressed-code record packaging** for `train_gpt.py`, `architectures.py`, and `configs.py`, which recovered artifact-size headroom without changing the trained model family.
6. **Sliding-window attention side path** present in-model, but submission authority remains the pulled `quantized_bpb` values above.

This record uses a fixed int6 model with **no TTT, no SLOT, no RLS, and no eval-time adaptation**. All three serialized artifacts are below the 16 MB cap. XSA telemetry is reported for completeness, but the submission authority remains `quantized_bpb`.

