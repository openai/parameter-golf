# Record: 11L Full GPTQ + Multi-Order N-gram Backoff Cache

**Fixed alpha: 3-seed mean val_bpb = 0.9757** (std=0.0002)
**Entropy-adaptive alpha: 3-seed mean val_bpb = 0.9605** (std=0.0003)

15.92 MB | 8xH100 SXM, 596s training + ~258s eval

## 3-Seed Results

| Seed | Neural-only | Fixed alpha (a=0.40) | Entropy-adaptive | Artifact | GPTQ Budget |
|------|-------------|----------------------|-----------------|----------|-------------|
| 1337 | 1.11719 | **0.97558** | **0.96027** | 15,921,027 B | 596s/600s |
| 42 | 1.11715 | **0.97562** | **0.96029** | 15,929,323 B | 596s/600s |
| 7 | 1.11787 | **0.97602** | **0.96082** | 15,922,059 B | 596s/600s |
| **Mean** | **1.11740** | **0.97574** | **0.96046** | | |
| **Std** | 0.00041 | 0.00024 | 0.00031 | | |

## Two Variants

### Variant 1: Fixed alpha (safest legal)

- `NGRAM_ENTROPY=0`, `NGRAM_ALPHA=0.40`
- Probability blend: `p = 0.60 * model + 0.40 * ngram` (constant for all tokens)
- **3-seed mean: 0.9757**

### Variant 2: Entropy-adaptive alpha

- `NGRAM_ENTROPY=1`, `alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))`
- Alpha depends on model output entropy H (high entropy = trust ngram more, low = trust model)
- Alpha is computed from the model's own output distribution ONLY, never from the true token
- **3-seed mean: 0.9605**

## Compliance

### Training (600s budget)
- Training stops at ~586s to reserve 14s for GPTQ calibration
- GPTQ Hessian calibration uses training data within the reserved budget
- Total: train 586s + GPTQ 10s = **596s** (within 600s)
- Budget check logged: `gptq:budget_check train:586069ms + gptq:9839ms = 595908ms (budget:600000ms)`
- **No training data is accessed during evaluation**

### Evaluation (600s budget)
- Sliding window eval (stride=64): ~86s
- N-gram cached eval (fixed alpha): ~130s
- N-gram cached eval (entropy-adaptive): ~140s
- Total eval (both variants): **~356s** (within 600s)

### N-gram Cache Legality
- **Backward-looking only**: n-gram count tables updated AFTER each window is scored, never before
- **No oracle selection**: probabilities are blended before seeing the true token; no min(NLL) picking
- **No training data access**: cache is built entirely from validation tokens during eval
- **Score-first protocol**: each sliding window's tokens are scored by the model first, then cache is updated
- **Fixed alpha variant**: constant alpha=0.40, no data-dependent weighting whatsoever
- **Entropy-adaptive variant**: alpha depends on model output distribution entropy only, never on the true token

### Artifact
- All seeds under 16,000,000 bytes (TARGET_MB=15.8 for seed 7)
- int6 per-row quantization + LZMA compression

## Architecture

11L, 512d, GQA 8H/4KV, LeakyReLU(0.5)^2 MLP 3x, XSA on all 11 layers, VE128, BigramHash(2048), Partial RoPE 16/64, LN Scale, SmearGate, U-Net skip connections, EMA(0.997), Parallel Muon optimizer, Full Hessian GPTQ int6.

## N-gram Cache Details

Multi-order backward-looking n-gram cache with backoff from order 7 down to order 2:
- For each token being scored, attempt the highest n-gram order first (7-gram context)
- If context has fewer than `min_count=2` occurrences in the cache, back off to a lower order
- Hash table: 4M buckets per order, XOR-shift hashing
- Cache update happens strictly AFTER scoring (backward-looking)

## Ablation (seed 1337)

| Configuration | val_bpb | Delta |
|---------------|---------|-------|
| Neural-only (no cache) | 1.1172 | baseline |
| Fixed 7-gram only, alpha=0.40 | 1.0258 | -0.0914 |
| Multi-order backoff (2-7), fixed alpha=0.40 | 0.9756 | -0.1416 |
| Multi-order backoff (2-7), entropy-adaptive | 0.9603 | -0.1569 |

## Credits

- Base architecture: PR #609 by @saml212 (XSA-all + selective pruning)
- Parallel Muon: PR #593 by @abaybektursun
- GPTQ budget fix: PR #535 by @raahilshah
- N-gram cache concept: PR #715, PR #727

## Run Command

```bash
# Fixed alpha variant (safest)
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=600 TARGET_MB=15.8 \
NGRAM_CACHE=1 NGRAM_ORDER=7 NGRAM_MIN_ORDER=2 NGRAM_ALPHA=0.40 \
NGRAM_MIN_COUNT=2 NGRAM_BUCKETS=4194304 NGRAM_ENTROPY=0 \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Entropy-adaptive variant
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=600 TARGET_MB=15.8 \
NGRAM_CACHE=1 NGRAM_ORDER=7 NGRAM_MIN_ORDER=2 NGRAM_ALPHA=0.40 \
NGRAM_MIN_COUNT=2 NGRAM_BUCKETS=4194304 NGRAM_ENTROPY=1 \
NGRAM_ENT_BASE=0.05 NGRAM_ENT_RANGE=0.55 NGRAM_ENT_SCALE=2.0 NGRAM_ENT_THRESH=4.0 \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
