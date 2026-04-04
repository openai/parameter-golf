## 04 — Experimental Results

This document consolidates the main BPB evidence into one place.

## A) Stable UT-era tuning (10-minute protocol)

Source logs:

- `logs/SWEEP_01_lr010_clip06.txt`
- `logs/SWEEP_02_lr011_clip08.txt`
- `logs/SWEEP_03_lr012_clip08.txt`
- `logs/SWEEP_04_tgn020.txt`
- `logs/SWEEP_05_tgn030.txt`

| Run | Matrix LR | Clip | Target Grad Norm | Final BPB |
|---|---:|---:|---:|---:|
| SWEEP_01_lr010_clip06 | 0.010 | 0.6 | 0.25 | 3.2051 |
| SWEEP_02_lr011_clip08 | 0.011 | 0.8 | 0.25 | 3.1952 |
| SWEEP_03_lr012_clip08 | 0.012 | 0.8 | 0.25 | 3.1918 |
| SWEEP_04_tgn020 | 0.012 | 0.8 | 0.20 | **3.1861** |
| SWEEP_05_tgn030 | 0.012 | 0.8 | 0.30 | 3.2002 |

Refinement follow-up:

| Run | Change | Final BPB | Outcome |
|---|---|---:|---|
| REFINE_01_baseline | baseline | 3.2075 | reference |
| REFINE_02_warmup12 | WARMUP_STEPS=12 | 3.3182 | worse |
| REFINE_03_muon4 | MUON_BACKEND_STEPS=4 | 3.2274 | worse |

## B) Full BPB journey snapshot

Source: `logs/bpb_full_journey.csv`

| Session | Run Label | Architecture label (from CSV) | Val BPB |
|---|---|---|---:|
| 1 | Initial_UT_12step | Universal Transformer | 3.75 |
| 1 | Best_3090_12step | Universal Transformer | 3.18 |
| 1 | dim512_discovery | Universal Transformer slim | 2.7335 |
| 2 | mlp2_steps2_LR012 | Single-pass Wide Transformer | **1.8085** |
| 2 | mlp5_steps1_lora512_BEST | Single-pass Wide Transformer | 1.8117 |

## C) AB3 feature sweep summary

Source: `results/ab3_sgbo_fixed/summary.csv`

| Run | SMEARGATE | BIGRAM_HASH | ORTHO_INIT | Best BPB |
|---|---:|---:|---:|---:|
| AB3_010 | 0 | 1 | 0 | **1.8279** |
| AB3_000 | 0 | 0 | 0 | 1.8374 |
| AB3_011 | 0 | 1 | 1 | 1.8492 |
| AB3_001 | 0 | 0 | 1 | 1.8550 |
| AB3_100 | 1 | 0 | 0 | 1.8729 |
| AB3_111 | 1 | 1 | 1 | 1.8826 |
| AB3_101 | 1 | 0 | 1 | 1.8866 |
| AB3_110 | 1 | 1 | 0 | 1.8874 |

## D) Invalid / non-comparable result tracking

The journey CSV explicitly marks:

- `SmearGate_BUG` (0.7690 BPB) as invalid due to non-causal leakage.

This result should never be treated as a valid model quality score.
