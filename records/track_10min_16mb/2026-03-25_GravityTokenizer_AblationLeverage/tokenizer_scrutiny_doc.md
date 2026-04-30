# Tokenizer Correctness Documentation

## Summary

The Gravity Tokenizer replaces 659 of 765 merge tokens in the standard 1024-vocabulary BPE tokenizer with tokens selected by ablation leverage scoring. The vocabulary size remains exactly 1024 (256 byte + 3 control + 765 merge). No tokens are added or removed — only which merge tokens occupy the 765 slots changes.

The val_bpb calculation uses the competition's own `build_sentencepiece_luts()` function with zero modifications. The gravity tokenizer is a standard SentencePiece Unigram model with byte fallback, compatible with the existing evaluation pipeline.

## The BPB Calculation Chain

The val_bpb metric is computed by the competition's evaluation function (`eval_val` in `train_gpt.py`). The calculation is tokenizer-agnostic by design:

### Step 1: Byte Count Lookup Tables

`build_sentencepiece_luts()` reads the SentencePiece `.model` file and builds three tensors:

- `base_bytes_lut[token_id]` — UTF-8 byte length of the token's surface form (excluding leading `▁` space marker)
- `has_leading_space_lut[token_id]` — whether the token begins with `▁`
- `is_boundary_token_lut[token_id]` — whether the token is control/unknown/unused

This function is **unmodified from the competition codebase**. It reads byte counts directly from the SentencePiece model proto via `sp.id_to_piece(token_id)` and `len(piece.encode("utf-8"))`.

### Step 2: Per-Token Byte Accounting During Eval

For every target token in the validation set:

```python
token_bytes = base_bytes_lut[target_id]
token_bytes += (has_leading_space_lut[target_id] & ~is_boundary_token_lut[prev_id])
```

This counts the exact number of UTF-8 bytes each token represents, including the space byte when appropriate. Every byte of the original text is accounted for exactly once.

### Step 3: BPB Computation

```python
val_loss = sum(per_token_losses) / total_tokens          # mean cross-entropy (nats)
bits_per_token = val_loss / ln(2)                         # convert nats to bits
tokens_per_byte = total_tokens / total_bytes              # tokenizer compression ratio
val_bpb = bits_per_token * tokens_per_byte                # bits per byte
```

## Why the Gravity Tokenizer is Penalized, Not Advantaged

The gravity tokenizer at beta=1.0 achieves 1.05 bytes/token, compared to 2.45 bytes/token for stock BPE. This means:

- **More tokens per byte of text** — the gravity tokenizer produces ~2.3x more tokens for the same content
- **Higher `tokens_per_byte` multiplier** — the BPB formula multiplies `bits_per_token` by `tokens_per_byte`, so a less-compressive tokenizer pays a direct penalty
- **The improvement is entirely in per-token prediction quality** — the model's cross-entropy per token must drop enough to more than compensate for the worse compression ratio

If there were a bug inflating byte counts (undercounting bytes per token), it would make `tokens_per_byte` smaller and **improve** BPB. But our tokenizer has shorter tokens than BPE, making `tokens_per_byte` larger, which **hurts** BPB. The direction of the compression ratio change works against us.

## Data Pipeline

### Retokenization

The validation and training data are retokenized through `retokenize_corpus.py`:

1. Load original BPE-tokenized shards (binary format, uint16 token IDs)
2. Decode to raw text using the original BPE SentencePiece model: `sp_base.decode(token_ids)`
3. Re-encode using the gravity SentencePiece model: `sp_gravity.encode(text)`
4. Write to new shards in identical binary format

The validation set is the same FineWeb first-50k-document split used by all competition entries. Only the tokenization changes.

### Tokenizer Construction

The gravity tokenizer is built via `build_tokenizer.py`:

1. Train a standard SentencePiece Unigram model to get the correct model proto structure
2. Replace the 765 merge token slots with gravity-selected tokens
3. Preserve all 256 byte-fallback tokens and 3 control tokens
4. Token scores (log-probabilities) are set from corpus frequency

The resulting `.model` file is a valid SentencePiece Unigram model with byte fallback. It uses `normalization_rule_name="identity"` and `character_coverage=1.0`, matching the competition baseline.

## Verification Checklist

| Check | Status |
|-------|--------|
| vocab_size == 1024 | Yes (3 control + 256 byte + 765 merge) |
| Byte fallback enabled | Yes |
| val set is identical FineWeb first-50k docs | Yes (same source shards, re-encoded) |
| `build_sentencepiece_luts()` unmodified | Yes (competition code, no changes) |
| `eval_val()` unmodified | Yes (competition code, no changes) |
| Roundtrip decode/encode preserves text | Yes (verified via spot checks) |
| Compression ratio penalizes gravity (1.05 vs 2.45 bytes/token) | Yes |
| int8+zlib quantization roundtrip evaluated | Yes (final_int8_zlib_roundtrip val_bpb reported) |
| 3-seed runs for statistical significance | Pending |

## Reproducibility

To reproduce the gravity tokenizer from scratch:

```bash
# 1. Generate extended candidate pool from BPE merge table
python scripts/generate_candidates_sp.py

# 2. Score all candidates by ablation leverage (requires GPU, ~4 hours)
python scripts/score_leverage.py

# 3. Build vocabulary with gravity scoring at beta=1.0
python scripts/build_vocabulary.py --beta 1.0

# 4. Build SentencePiece model from gravity vocabulary
python scripts/build_tokenizer.py \
    --vocabulary data/vocabularies/vocabulary_beta_1.0.json \
    --output data/tokenizers/gravity_beta_1.0.model \
    --corpus-sample data/corpus_sample.txt

# 5. Retokenize FineWeb shards
python scripts/retokenize_corpus.py \
    --base-tokenizer parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
    --gravity-tokenizer data/tokenizers/gravity_beta_1.0.model \
    --data-dir parameter-golf/data/datasets/fineweb10B_sp1024 \
    --output-dir parameter-golf/data/datasets/fineweb_gravity_beta_1.0

# 6. Train (identical to competition baseline, just different data path + tokenizer path)
DATA_PATH=./data/datasets/fineweb_gravity_beta_1.0 \
TOKENIZER_PATH=./data/tokenizers/gravity_beta_1.0.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The scoring pipeline (steps 1-2) requires ~4 hours on a single GPU. Steps 3-5 are deterministic and complete in minutes. The training script is the competition's `train_gpt.py` with only `DATA_PATH` and `TOKENIZER_PATH` changed.
