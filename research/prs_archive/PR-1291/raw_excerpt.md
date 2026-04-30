# PR 1291 — Vocab4096 + MLP4.0x + SLOT

**Author:** Nathan Maine (dentity007)
**Claimed BPB:** 1.0925 (3-seed mean, std 0.0018; seeds 42, 1337, 2025; per-seed 1.0947, 1.0913, 1.0915)
**Artifact size:** ~15.95 MB (per-seed: 15,954,746 / 15,932,192 / 15,948,156 bytes)
**Seeds:** 42, 1337, 2025
**Track:** 10min_16mb
**Base PR:** 1218

## Files retrieved
- `APPROACH.md`
- `records__track_10min_16mb__2026-04-03_Vocab4096_MLPMult4_SLOT_1.0925__README.md`
- `records__track_10min_16mb__2026-04-03_Vocab4096_MLPMult4_SLOT_1.0925__submission.json`
- `records__track_10min_16mb__2026-04-03_Vocab4096_MLPMult4_SLOT_1.0925__train_gpt.py`

## Environment variables (from README run command)
SEED=42, SLOT_ENABLED=1, SLOT_LR=0.005, SLOT_STEPS=8

## Claimed changes (from README, verbatim)

> Built on PR #1218 (@clarkkev) with SLOT eval-time optimization added.

> 11L transformer, d=512, 8H/4KV GQA, MLP 4.0x; Vocabulary 4096 (sp4096 tokenizer); XSA all 11 layers, QK_GAIN=4.0; EMA 0.997, dynamic warmdown 66.7%; Muon WD=0.085, embeddings WD=0.085, LR=0.02; Sigmoid-gated U-Net skip; 34.4M parameters.

> SLOT: Per-Batch Delta Optimization. After sliding window evaluation, SLOT optimizes a small additive delta vector at the last hidden layer: 1. forward_hidden(): Compute hidden states under no_grad() (frozen transformer). 2. Optimize delta: 8 AdamW steps (lr=0.005) through compute_logits() only. 3. Score: Final logits computed with optimized delta, full softmax distribution. Delta shape [1, 1, 512], re-initialized to zeros per batch. SLOT contribution: -0.0067 to -0.0069 BPB across seeds.

> Legality: SLOT is score-first (hidden states under no_grad before any optimization); delta re-initialized per batch; no TTT; no n-gram cache.

> Full Hessian GPTQ with AR self-generated calibration; Int6 + byte shuffle + brotli-11.
