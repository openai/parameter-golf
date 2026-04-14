# PR 1302 — Split-LR + N-gram Agreement + Full Hessian GPTQ + Brotli

**Author:** vlivashkin
**Claimed BPB:** 1.1078 (3-seed mean, std 0.0009; val_loss 1.87521 nats; seeds 1337, 42, 2025; per-seed 1.1083, 1.1068, 1.1085)
**Artifact size:** ~15.86 MB (per-seed: 15,853,466 / 15,857,705 / 15,846,914)
**Seeds:** 1337, 42, 2025
**Track:** 10min_16mb
**Delta vs SOTA (PR #1019):** -0.00697 nats

## Files retrieved
- `records__track_10min_16mb__2026-04-03_SplitLR_NgramAgreement_FullGPTQ__README.md`
- `records__track_10min_16mb__2026-04-03_SplitLR_NgramAgreement_FullGPTQ__submission.json`
- `records__track_10min_16mb__2026-04-03_SplitLR_NgramAgreement_FullGPTQ__train_gpt.py`

## Environment variables (from README run command)
BIGRAM_DIM=160, SEED=1337|42|2025 (training); CHECKPOINT=checkpoints/final_model_seed${SEED}.int6.ptz, BIGRAM_DIM=160 (ngram eval)

## Claimed changes (from README, verbatim)

> What's New vs PR #1019. Training improvements (from PR #1179): 1. Split-LR — different learning rates for early (0.025) vs late (0.030) layers. 2. BigramHash(2816x160) — wider projection (160 vs 112), fewer buckets. 3. Sigmoid-gated U-Net — learnable gates on encoder-decoder skip connections. 4. Soft-round QAT — temperature-controlled rounding (alpha 1->16) replacing STE. 5. Brotli-11 + byte-shuffle — saves ~400KB vs LZMA. 6. Coprime-stride data loader. Evaluation improvement (from PR #1145): 7. Online n-gram agreement — 3 causal experts (token n-gram, within-word, word-start). Contributes -0.0028 BPB.

> Three online n-gram experts predict the next token using only already-scored (past) tokens: token n-gram (16-gram context, hash table), within-word continuation, word-start hints. When 2+ experts agree, boost is increased. LLM probability adjusted via exponential tilting: p_adjusted = (scale * p_true) / (1 - p_hint + scale * p_hint). Causal, score-first (torch.inference_mode()), properly normalized.

> Architecture: 11 layers (512d, 8 GQA heads, 4 KV heads), MLP 3x (1536) LeakyReLU(0.5)^2, XSA on all 11, BigramHash 2816x160, Split-LR early=0.025 late=0.030 bank_split=5, sigmoid-gated U-Net skips, soft-round QAT, partial RoPE 16/64, LN scale 1/sqrt(layer+1), VE128 layers 9-10, SmearGate, EMA(0.997)+SWA(every 50), full Hessian GPTQ int6, Brotli-11 + byte-shuffle, Parallel Muon + Parameter Banking. Eval time ~449s (n-gram) of 536s total.
