# GDN-Hybrid + Sliding Window Attention (3-seed mean 1.02045733 BPB)

Joshua-owned SAFE_SUBMISSION reproduction of upstream PR #1545, reconciled from pulled TensorPool artifacts for `run037-safe017` / `j-jvquftkrwd`.

## Headline result

- **SAFE_SUBMISSION authority:** `quantized_bpb`
- **3-seed mean:** **1.02045733 BPB**
- **3-seed std:** **0.00553017 BPB**
- **Best seed:** **1.016586 BPB**
- **Worst seed:** `1.026791 BPB`
- **Artifact size range:** `15,313,984` to `15,830,308` bytes
- **Legality lane:** **SAFE_SUBMISSION** — fixed-predictor / no-TTT Track-A; all pulled artifacts stayed below the `16,000,000` byte cap

## Per-seed authoritative results

| Seed | Steps | EMA BPB | Quantized BPB | XSA BPB | Artifact bytes |
|------|------:|--------:|--------------:|--------:|---------------:|
| 42 | 1864 | 1.017723 | 1.026791 | 1.031731 | 15,313,984 |
| 1337 | 2239 | 1.007375 | 1.016586 | 1.020691 | 15,830,308 |
| 2024 | 2241 | 1.008736 | 1.017995 | 1.023138 | 15,820,201 |
| **Mean** | — | **1.011278** | **1.02045733** | **1.025187** | **15,654,831.00** |
| **Std (sample)** | — | — | **0.00553017** | — | — |

## Why this matters

- Improves Joshua's staged fork submission branch `submission-run036-safe016-1.0585` by **0.03804398 BPB**.
- Materially outperforms the previously staged Joshua-owned SAFE_SUBMISSION artifact (`run036-safe016`, 1.05850131 BPB).
- Reproduces the strongest visible clean-lane architecture on Joshua-owned infrastructure using pulled artifacts as authority.

## Technique stack

1. **SP1024 tokenizer** with a GDN-hybrid backbone (`[GDN×5] → SWA → [GDN×5] → SWA_shared`).
2. **Fixed-predictor / no-TTT Track-A lane** — no eval-time or pre-quant adaptation in the scored artifact.
3. **MuonEq-R + AdamW** training mix, EMA `0.997`, late QAT threshold `0.15`.
4. **GPTQ int6 + zstd-22** packaging.
5. **Sliding-window attention side path** present in-model, but submission authority remains the pulled `quantized_bpb` values above.

## Legality notes

This record is **SAFE_SUBMISSION** because the scored artifact is a fixed int6 model with **no TTT, no SLOT, no RLS, and no eval-time adaptation**. The pulled logs show all three serialized artifacts below the 16 MB cap. XSA telemetry is reported for completeness, but the submission authority for this lane remains `quantized_bpb` from the pulled artifacts.

## Provenance

- **TensorPool job:** `j-jvquftkrwd`
- **Canonical bundle:** `~/parameter-golf-project/jobs/run037-safe017/`
- **Pulled artifacts:** `~/parameter-golf-project/state/tp-pulls/run037-safe017/artifacts/`
- **Upstream source PR:** `openai/parameter-golf#1545`
