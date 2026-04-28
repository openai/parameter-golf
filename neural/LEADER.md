# Neural SOTA — Current Leader

Score:  1.10986874 BPB (seed 444) | 1.1099 mean (3-seed)
Size:   15.44MB
Date:   2026-03-30
Leg:    neural/2026-03-30_Rascal_II/
Hash:   0ec1f462ab39fd601b18f2b086f6283a0c8db3d2a9780a92dfb206ec46e067cb
Run:    bash scripts/sota_now.sh

## Architecture
Junkyard Rat Rascal II — 11L XSA-all + Parallel Muon + Coprime loader
Bigram2048 + RoPE16 + SWA (step ~5900) + Late QAT (step ~6070, scale=0.15)
SKIP_GPTQ=1 | naive int6 (5 layers + embed) | zstd compressed
26.99M params | 6593 steps @ ~91ms/step on 8xH100

## Seeds
| Seed | BPB exact       | Size          |
|------|-----------------|---------------|
| 42   | 1.11018163      | 15,540,001 B  |
| 300  | 1.10979099      | 15,542,719 B  |
| 444  | 1.10986874      | 15,554,053 B  |
| mean | **1.1099**      | 15.44MB       |

## Promotion Gate
Beat 1.10986874 on seed 444 → confirm on seed 300 → update this file.
One variable changed per leg. Gate (1-GPU, 2000 steps) before any 8x run.
