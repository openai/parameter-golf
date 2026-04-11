# Record: Score-First TTT + Multi-Order N-gram Backoff (val_bpb=0.9581)

**3-seed mean val_bpb: 0.9581** (std=0.0005) | ~15.7 MB artifact | 8xH100 SXM

## Results

| Seed | Sliding BPB (s64) | Artifact | Steps | ms/step | TTT time | Total eval |
|------|-------------------|----------|-------|---------|----------|------------|
| 1337 | 0.9576 | 15,721,728 | 6409 | 93.63 | 107.0s | ~303s |
| 42 | 0.9581 | 15,702,393 | 6403 | 93.73 | 107.9s | ~255s |
| 7 | 0.9585 | 15,768,158 | 6407 | 93.65 | 105.2s | ~251s |
| **Mean** | **0.9581** | | ~6406 | ~93.67 | ~106.7s | ~270s |

## Architecture

- 11L, 512d, GQA (8H/4KV), MLP 3x, U-Net skip connections
- LeakyReLU(0.5)^2: preserves negative gradient flow
- XSA on all 11 layers: removes self-position bias
- Value Residual (VR): layer 0 V output mixed via sigmoid gates
- Gated Attention (GA): per-head sigmoid gates
- SmearGate + OrthoInit, BigramHash(4096), Partial RoPE (16/64), LN Scale
- EMA(0.997), warmdown=3000, int6 per-row + zstd-16

## Eval-Time Techniques

### Score-First TTT (compliant with Issue #677)
- Process val data in sequential 131K-token chunks
- Phase 1: Score chunk under inference_mode (forward only)
- Phase 2: Train on scored tokens with AdamW (lr=0.0001, 4 epochs)
- Freeze first 2 blocks, grad clip 1.0
- Each token scored BEFORE model trains on it

### Multi-Order N-gram Backoff + Entropy-Adaptive Alpha
- Orders 2-7: highest order first, cascade on miss
- Entropy-adaptive: alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))
- Fixed formula, no oracle selection, no target-aware gating
- Backward-looking: cache built from already-scored tokens only

## Compliance

- Score-first TTT: tokens scored under inference_mode before training
- N-gram cache: backward-looking, entropy-based mixing (not target-aware)
- GPTQ: not used (naive int6 per-row quantization)
- All training within 600s, all eval within 600s
- No training data accessed at eval time

## Reproduction

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
SEED=1337 TTT_ENABLED=1 NGRAM_CACHE=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- Base: modded-nanogpt, PR #315, #609
- LeakyReLU^2: PR #493, #518
- Value Residual: PR #413 (arXiv:2410.17897)
- Gated Attention: NeurIPS 2025 (arXiv:2505.06708)
- N-gram cache concept: PR #659, #702
- Score-first TTT: PR #549
