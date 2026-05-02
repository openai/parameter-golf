---
name: parameter-golf competition state
description: Current state of the parameter-golf competition entry — 14L model at 1.1130 BPB, best clean score, key findings and next steps
type: project
---

## Competition: OpenAI Parameter Golf
Train best LM in 16MB artifact + 10min train + 10min eval on 8xH100, scored by BPB on FineWeb val.

## Our Best: 1.1130 val_bpb (best clean score on the repo)
- 14-layer, 512d, 8H/4KV GQA, MLP 3x leaky_relu(0.5)², FlashAttention 3
- EMA(0.997), Muon WD=0.09, GPTQ int6, brotli-11 compression
- Online per-window SGD TTT (score-first, legal per issue #402), stride=76
- ~105ms/step, 5700 steps in 600s, 15.83MB artifact, 575s eval time

## Key Finding: We are at critical depth (14 of ~16 Dcrit for dim=512)
- arxiv:2601.20994 "Depth Delusion": Dcrit ∝ W^0.44, at W=512 → Dcrit≈16
- DO NOT add more layers. 15L was already worse.
- Techniques that work at 11L (XSA, LN Scale, Partial RoPE) fail at 14L because we're near the gradient starvation cliff.

## #1 Problem: Quantization gap is 0.015 BPB (2x competitors' 0.007)
- Root cause: GPTQ calibrates with ideal float activations, but inference uses quantized outputs. Error compounds over 14 layers.
- QEP (arxiv:2504.09629, NeurIPS 2025) fixes this by propagating quantized outputs during calibration.
- Expected gain: 0.003-0.007 BPB. This is the highest-priority implementation.

## AB_TESTS_ROUND3.md is the source of truth
- All experiments, research, sources, and results tracked there
- Must be updated before/after every A/B test and when new research is found

## Server
- SSH: `ssh -p 7122 rganapa@animal.netravi.net` (password: Summer02!)
- Scripts at: `/data/backups/rganapa/parameter-golf/`
- Saved pre-TTT model: `final_model_pre_ttt.pt` (for eval-only TTT experiments)
- wandb project: `parameter-golf` (entity: ishanramrakhiani-bindwell)

## What's been tried and failed at 14L
- XSA (both orthogonal and mean-subtract): hurt or leaked info
- Value Embeddings (PR #505 style): hurt when combined with XSA
- LN Scale: +0.028 worse
- Partial RoPE: worse
- Late QAT: made weights less compressible
- Butterfly/FFT layers: 2.6x slower
- Temperature scaling T=0.98 on TTT: +0.0002 worse
- Chunked AdamW TTT: 1.1175 (0.005 worse than per-window SGD)

**Why:** Confirmed by Depth Delusion paper — at 14/16 critical depth, perturbations to the gradient flow are amplified.
