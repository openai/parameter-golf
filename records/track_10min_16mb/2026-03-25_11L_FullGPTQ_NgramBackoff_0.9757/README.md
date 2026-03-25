# Record: 11L Full GPTQ + Multi-Order N-gram Backoff Cache

**3-seed mean val_bpb = 0.9757** (std=0.0002) | 15.92 MB | 8xH100 SXM, 600s training + 258s eval

## 3-Seed Results

| Seed | Steps | ms/step | Neural-only BPB | N-gram BPB (fixed alpha) | Artifact | GPTQ Budget |
|------|-------|---------|-----------------|--------------------------|----------|-------------|
| 1337 | ~6500 | 87 | 1.1172 | **0.9756** | 15,929,323 B | 596s/600s |
| 42 | ~6500 | 87 | 1.1171 | **0.9756** | 15,929,323 B | 596s/600s |
| 7 | ~6500 | 87 | 1.1179 | **0.9760** | 15,922,059 B | 596s/600s |

**Mean: 0.9757 | Std: 0.0002**

## Compliance

### Training (600s budget)
- Training stops at ~586s to reserve 14s for GPTQ calibration
- GPTQ Hessian calibration uses training data (`fineweb_train_*`) within the reserved budget
- Total: train 586s + GPTQ 10s = **596s** (within 600s)
- Budget check logged: `gptq:budget_check train:586069ms + gptq:9839ms = 595908ms (budget:600000ms)`
- No training data is accessed during evaluation

### Evaluation (600s budget)
- Sliding window eval (stride=64): ~86s
- N-gram cached eval: ~130s
- Total eval: **~258s** (within 600s)

### N-gram Cache Legality
- **Backward-looking only**: n-gram count tables are updated AFTER each window is scored, never before
- **Fixed-weight blend**: `mixed = (1 - 0.40) * model_prob + 0.40 * ngram_prob` — alpha is constant, not adaptive
- **No oracle selection**: probabilities are blended before seeing the true token; no min(NLL) picking
- **No training data access**: cache is built entirely from validation tokens during eval
- **Score-first protocol**: each sliding window's tokens are scored by the model first, then the n-gram tables are updated with those tokens for future windows

### Artifact
- All seeds under 16,000,000 bytes
- int6 per-row quantization + LZMA compression

## Architecture

11L, 512d, GQA 8H/4KV, LeakyReLU(0.5)^2 MLP 3x, XSA on all 11 layers, VE128, BigramHash(2048), Partial RoPE 16/64, LN Scale, SmearGate, U-Net skip connections, EMA(0.997), Parallel Muon optimizer, Full Hessian GPTQ int6.

## N-gram Cache Details

Multi-order backward-looking n-gram cache with backoff from order 7 down to order 2:
- For each token being scored, attempt the highest n-gram order first (7-gram context)
- If context has fewer than `min_count=2` occurrences in the cache, back off to a lower order
- Probability blend: `p_mixed = 0.60 * p_model + 0.40 * p_ngram`
- Hash table: 4M buckets per order, XOR-shift hashing
- Cache update happens AFTER scoring (backward-looking)

## Ablation (seed 1337)

| Configuration | val_bpb | Delta |
|---------------|---------|-------|
| Neural-only (no cache) | 1.1172 | baseline |
| Fixed 7-gram, alpha=0.40 | 1.0258 | -0.0914 |
| **Multi-order backoff (2-7), alpha=0.40** | **0.9756** | **-0.1416** |

## Credits

- Base architecture: PR #609 by @saml212 (XSA-all + selective pruning)
- Parallel Muon: PR #593 by @abaybektursun
- GPTQ budget fix: PR #535 by @raahilshah (reserved calibration time within training budget)
- N-gram cache concept: PR #715, PR #727

## Run Command

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TARGET_MB=15.8 \
NGRAM_CACHE=1 \
NGRAM_ORDER=7 \
NGRAM_MIN_ORDER=2 \
NGRAM_ALPHA=0.40 \
NGRAM_MIN_COUNT=2 \
NGRAM_BUCKETS=4194304 \
NGRAM_ENTROPY=0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
