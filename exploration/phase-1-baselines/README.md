# Phase 1: Baseline Improvements

**Dates:** Mar 17-22, 2026
**Goal:** Push the upstream 9-layer 512-dim transformer as far as possible before exploring new architectures.
**Outcome:** Achieved 1.1233 BPB — SOTA on the leaderboard through pure training optimization.

## Submissions (chronological)

| Date | Name | Key Changes | BPB |
|------|------|-------------|-----|
| Mar 17 | NaiveBaseline | Starting point | 1.2244 |
| Mar 17 | LoRA_TTT | Test-time training with LoRA | — |
| Mar 18 | FP16Embed_WD3600 | FP16 embeddings, longer warmdown | — |
| Mar 18 | LongContextSeq2048 | Sequence length 2048 | — |
| Mar 18 | LowerLR | Learning rate reduction | — |
| Mar 19 | Various | Int6 QAT, SwiGLU, sliding window, mixed quant | — |
| Mar 20 | SmearGate+BigramHash | Local context injection at embedding layer | — |
| Mar 20 | 11L_XSA4_EMA | 11 layers, cross-skip attention, EMA | 1.1271 |
| Mar 21 | PartialRoPE_LateQAT | Partial RoPE, delayed QAT start | 1.1248 |
| Mar 22 | EMA_GPTQ-lite | EMA + GPTQ-lite calibration | **1.1233** |
| Mar 23 | LeakyReLU_LegalTTT | LeakyReLU, legal TTT, parallel Muon | — |

## Key Findings

- Sliding window evaluation was a free win for BPB
- SmearGate + BigramHash gave the model cheap local context without extra parameters
- EMA model averaging during warmdown consistently helped
- Int6 QAT with STE was better than post-training quantization
- 11 layers at 512-dim was the sweet spot for the 16MB budget on a transformer

## What Led to Phase 2

With the baseline transformer near its ceiling, the next question was: can we get more from the same parameter budget with architectural changes? Multiskip connections were the first experiment.
