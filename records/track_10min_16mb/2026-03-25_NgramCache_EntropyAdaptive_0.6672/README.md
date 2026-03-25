# 11L + Multi-Order N-gram Backoff + Entropy-Adaptive Alpha

**val_bpb: 0.6672** (seed 42) | **15.0 MB** artifact | 1xB200 (HiPerGator)

## Technique

Base 11L SOTA architecture with a novel eval-time n-gram cache that provides -0.49 BPB improvement over neural-only sliding eval.

### Multi-order N-gram Backoff (orders 2-7)

During sliding window evaluation, we maintain hash tables for n-gram contexts of orders 2 through 7. For each token prediction, we attempt the highest order first and cascade down on miss. This captures repeated patterns within documents that the neural model cannot access outside its context window.

### Entropy-Adaptive Alpha

Instead of a fixed interpolation weight, alpha adapts based on the model's own entropy:

```
alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))
```

- Low entropy (model confident): alpha -> 0.05, trust the LM
- High entropy (model uncertain): alpha -> 0.60, trust the n-gram cache

### Compliance

- Score-first, backward-looking: n-gram counts built from previously scored tokens only
- No oracle selection: alpha depends on model entropy, never on ground-truth labels
- Single blended prediction per token, no min(NLL)

## Results

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1927 |
| Post-quant roundtrip | 1.1577 |
| **Post n-gram sliding (s64)** | **0.6672** |
| Artifact size | 15,025,238 bytes |
| Training steps | 20,000 |
| Step avg | 538.9 ms |

## Architecture

- 11L, 512d, 8H/4KV GQA, MLP 3x
- XSA last 4 layers, Partial RoPE (16/64), LN Scale
- Value Embeddings (VE128, layers 9-10)
- SmearGate + BigramHash(2048)
- EMA (0.997), Late QAT (0.15), OrthoInit
- Int6 per-row + GPTQ-lite + 3% magnitude pruning + zstd-22

## Reproduction

```bash
pip install sentencepiece zstandard
python3 data/cached_challenge_fineweb.py --variant sp1024

SEED=42 NGRAM_CACHE=1 NGRAM_ORDER=7 NGRAM_MIN_ORDER=2 \
NGRAM_ENTROPY=1 EVAL_STRIDE=64 PRUNE_PCT=0.03 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Credits

- Base architecture: PR #414 (signalrush), PR #315 (jfprincz), PR #287 (jfprincz)
- N-gram cache concept: PR #702 (lukacf), PR #727 (lukacf)
- Entropy-adaptive alpha: PR #727 (lukacf)
