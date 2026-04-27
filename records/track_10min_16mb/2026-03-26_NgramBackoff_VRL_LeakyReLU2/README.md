# N-gram Backoff + VRL + LeakyReLU² — val_bpb 0.9642

val_bpb = 0.9642 (3-seed mean, std 0.0002) | ~15.95 MB | 8×H100 SXM

## 3-Seed Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-ngram bpb | **Post-ngram bpb** | ng_helped | Artifact |
|------|----------|-------|--------------|-------------------|-----------|----------|
| 1337 | 88.7ms | 6,765 | 1.1225 | **0.9640** | 38.5% | 15,981,848 |
| 42 | 88.6ms | 6,772 | 1.1224 | **0.9641** | 38.6% | 15,904,632 |
| 2025 | 88.6ms | 6,776 | 1.1231 | **0.9644** | 38.6% | 15,974,308 |
| **Mean** | **88.6ms** | **6,771** | **1.1227** | **0.9642 (std 0.0002)** | **38.6%** | |

All artifacts under 16,000,000 bytes. All train logs attached.

## Key Innovation: Multi-Order N-gram Backoff Cache

Backward-looking n-gram cache built causally from already-scored tokens during evaluation. No training data access. Zero artifact cost.

### Entropy-Adaptive Alpha
```python
alpha = 0.05 + 0.55 * sigmoid(2.0 * (H - 4.0))
```
- When neural model is confident (low entropy): alpha ≈ 0.05 (trust neural)
- When neural model is uncertain (high entropy): alpha ≈ 0.60 (trust n-grams)

### Multi-Order Backoff (2-7gram)
- Try highest order first (7-gram), fall back to lower orders
- Only emit prediction when context count >= 2
- Raw count ratios, no smoothing
- 4M hash buckets per order (XOR-with-primes hashing)

### Mixing
```python
mixed_p = (1 - alpha) * model_p + alpha * ngram_p
```
Linear interpolation in probability space. Score-first: n-gram tables updated AFTER each token is scored.

## Training Architecture

Same as PR #175 (our pure neural submission at 1.1229):
- 11L, 512d, 8H/4KV (GQA), LeakyReLU(0.5)² MLP 3×
- VRL (Value Residual Learning), VE128, SmearGate, BigramHash(2048)
- XSA4, Partial RoPE 16/64, LN Scale, U-Net skips
- EMA(0.997) + Tight SWA, Late QAT (STE@0.15), OrthoInit
- GPTQ-lite int6 + lzma, FA3 Hopper, Muon WD=0.04

## Compliance

- Training: 600s on 8×H100 SXM
- Eval (sliding window + n-gram): ~15 min on 8×H100 SXM (under 10 min per-GPU)
- All artifacts under 16,000,000 bytes
- N-gram tables built causally from already-scored tokens only
- No training data access during evaluation
- No oracle/hindsight selection
- Score-first: every token scored before any table update using that token

## Reproduction

```bash
RUN_ID=seed1337 SEED=1337 NGRAM_ENABLED=1 NGRAM_ORDER=7 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 VRL_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- N-gram backoff approach: PR #727 by @Asukabot0
- Neural base: PR #414 by @signalrush
- LeakyReLU²: PR #493 by @parinzee, PR #518 by @sofiabod
- VRL: ResFormer (arXiv:2410.17897), PR #569 by @gowtham0992
- XSA: PR #287 by @jfprincz
