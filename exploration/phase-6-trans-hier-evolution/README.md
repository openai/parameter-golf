# Phase 6: Trans-Hier Evolution

**Dates:** Mar 26, 2026
**Goal:** Take the macro-sidechannel concept from Phase 5 and build a full competitive architecture: the Transformer-Hierarchical (trans-hier) model.
**Outcome:** From concept to ~1.18 BPB. The architecture combined encoder-decoder split, macro sidechannel distillation, and int4 QAT into a submission-ready model.

## Runs

| Run | Stage | Notes |
|-----|-------|-------|
| 008-trans-hier-sidechannel | v1 | First integration of sidechannel into transformer |
| 009-distill-chunk32-dualpass | v2 | Dual-pass distillation with chunk size 32 |
| 010-trans-hier-dualdistill | v3 | Refined dual distillation |
| 011-trans-hier-int4qat | v4 | Added int4 QAT — submission-ready (has PROVENANCE.md) |
| 012-trans-hier-int4qat-9L-8gpu | v5 | 9 layers, full 8xH100 run |
| trans-parallel | alt | Parallel variant exploration |

## Architecture at This Stage

- Encoder-Decoder split (5+5 layers) with U-Net skip connections
- Macro sidechannel: causal cross-attention at encoder/decoder boundary
- Macro pyramid: two-level hierarchy (interval=16, interval=64)
- Int4 QAT from step 1 with STE
- SmearGate + BigramHash at embedding layer

## What Led to Phase 7

With trans-hier working, Phase 7 explored alternative architectures (r-series) to ensure we weren't stuck in a local optimum. Meanwhile, the SSD track (Phase 8) continued in parallel.
