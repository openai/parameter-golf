# GDN-Hybrid + Sliding Window Attention (cold-cache 3-seed mean 1.01710033 BPB)

Joshua-owned SAFE_SUBMISSION confirmation run for the GDN-Hybrid family, reconciled from pulled TensorPool artifacts for `run039-safe019` / `j-xpv7d0b38j`.

## Headline result

- **SAFE_SUBMISSION authority:** `quantized_bpb`
- **3-seed mean:** **1.01710033 BPB**
- **3-seed std:** **0.00133490 BPB**
- **Best seed:** **1.016192 BPB**
- **Worst seed:** `1.018633 BPB`
- **Artifact size range:** `15,522,111` to `15,981,262` bytes
- **Legality lane:** **SAFE_SUBMISSION** — fixed-predictor / no-TTT Track-A; all pulled artifacts stayed below the `16,000,000` byte cap

## Per-seed authoritative results

| Seed | Steps | EMA BPB | Quantized BPB | XSA BPB | Artifact bytes |
|------|------:|--------:|--------------:|--------:|---------------:|
| 314 | 2223 | 1.007670 | 1.016476 | 1.020950 | 15,522,111 |
| 777 | 2239 | 1.007192 | 1.016192 | 1.020919 | 15,814,260 |
| 2718 | 2240 | 1.009535 | 1.018633 | 1.023874 | 15,981,262 |
| **Mean** | — | **1.008132** | **1.01710033** | **1.021914** | **15,772,544.33** |
| **Std (sample)** | — | — | **0.00133490** | — | — |

## Why this matters

- Improves Joshua's prior staged 3-seed SAFE_SUBMISSION artifact `run037-safe017` (**1.02045733 BPB**) by **0.00335700 BPB**.
- Confirms the GDN-Hybrid family remains strong under a fresh cold-cache 3-seed confirmation bundle.
- Reproduces the strongest visible clean-lane architecture family on Joshua-owned infrastructure using pulled artifacts as authority.

## Technique stack

1. **SP1024 tokenizer** with a GDN-hybrid backbone (`[GDN×5] → SWA → [GDN×5] → SWA_shared`).
2. **Fixed-predictor / no-TTT Track-A lane** — no eval-time or pre-quant adaptation in the scored artifact.
3. **MuonEq-R + AdamW** training mix, EMA `0.997`, late QAT threshold `0.15`.
4. **GPTQ int6 + zstd-22** packaging.
5. **Sliding-window attention side path** present in-model, but submission authority remains the pulled `quantized_bpb` values above.

## Legality notes

This record is **SAFE_SUBMISSION** because the scored artifact is a fixed int6 model with **no TTT, no SLOT, no RLS, and no eval-time adaptation**. The pulled logs show all three serialized artifacts below the 16 MB cap. XSA telemetry is reported for completeness, but the submission authority for this lane remains `quantized_bpb` from the pulled artifacts.

## Provenance

- **TensorPool job:** `j-xpv7d0b38j`
- **Canonical bundle:** `~/parameter-golf-project/jobs/run039-safe019/`
- **Pulled artifacts:** `~/parameter-golf-project/state/tp-pulls/run039-safe019/artifacts/`
- **Upstream source PR:** `openai/parameter-golf#1545`
