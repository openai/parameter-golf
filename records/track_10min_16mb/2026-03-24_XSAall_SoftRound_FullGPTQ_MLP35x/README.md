# Record: XSA-all + Soft-Round QAT + Full GPTQ + MLP 3.5x + AdamW TTT

## Summary

Combines six proven improvements from pending PRs #606 and #609 onto the official SOTA (#549) base:

1. **XSA on all 11 layers** (from #609): Forces cross-position information mixing from layer 0. -0.0016 BPB vs XSA-4. Zero new parameters.

2. **Soft-Round QAT** (from #606): Replaces STE rounding with differentiable `s_α(y) = floor(y) + 0.5 * tanh(α·(frac-0.5)) / tanh(α/2) + 0.5`. Alpha anneals from 1→16 during QAT phase. Better gradient flow than STE at zero cost.

3. **Full GPTQ** (from #609/#606): Hessian-aware column-reordered quantization with error redistribution. Tries 5 clip percentiles per row. Replaces naive int6 quantization for all large weight matrices.

4. **MHA 8/8 + MLP 3.5x** (from #606): Full multi-head attention (8 heads / 8 KV heads) with 3.5x MLP expansion (1792 hidden). Enables ~33.6M parameter model under 16MB via int6 compression.

5. **BigramHash(8192)** (from #606): 8192-bucket bigram hash table (vs 2048 baseline). Reduces token-pair collisions.

6. **AdamW TTT** (from #606): Replaces SGD with AdamW for legal score-first TTT. 131K-token chunks, cosine LR decay, all blocks unfrozen.

7. **Selective ±1 magnitude pruning** (from #609): Post-GPTQ, zeros the smallest-impact ±1 quantized values to fit artifact under target size.

Everything else from #549 carries forward: 11L, 512d, LeakyReLU(0.5)², Partial RoPE 16/64, LN Scale, VE128 shared (layers 9-10), SmearGate, U-Net skips, EMA(0.997), Tight SWA, Parameter Banking + Parallel Muon, LZMA compression.

## Architecture

- 11 layers, model_dim=512, 8 heads / 8 KV heads (MHA), MLP 3.5x (1792)
- XSA on all 11 layers, Partial RoPE 16/64, LN Scale
- SmearGate + OrthoInit, BigramHash(8192), Shared VE128 (layers 9-10)
- LeakyReLU(0.5)² activation
- U-Net skip connections (5 encoder + 6 decoder)

## Quantization

- Soft-Round QAT during training (alpha 1→16 annealing)
- Full GPTQ int6 per-row (31 levels) with Hessian-aware error compensation
- Selective ±1 magnitude pruning post-GPTQ
- LZMA-6 compression
- Target: ≤15.9 MB artifact

## TTT (Legal, Score-First)

- AdamW optimizer (lr=0.0001, wd=0.0)
- 131,072-token chunks, 3 epochs/chunk, cosine LR decay
- All blocks unfrozen
- Strictly backward-looking: score FIRST, then train

## Run command

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

SEED=1337 TTT_ENABLED=1 TARGET_MB=15.9 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-24_XSAall_SoftRound_FullGPTQ_MLP35x/train_gpt.py
```

## Requirements

- Flash Attention 3 (Hopper kernel): `from flash_attn_interface import flash_attn_func`
- `zstandard`, `sentencepiece`, `lzma` (stdlib)

## Credits

- Base model + Parallel Muon: PR #549 by @sanjeevmadhav / @abaybektursun
- Full GPTQ: PR #535 by @raahilshah, PR #569 by @gowtham0992
- Soft-Round QAT + int5 GPTQ + AdamW TTT: PR #606 by @EthanYangTW
- XSA-all + Selective Pruning: PR #609 by @saml212
- LeakyReLU(0.5)²: PR #493, #518
- XSA: arXiv:2603.09078
