# Custom Tokenizer for Parameter Golf: Web-Content Symbols + split_digits=false

**Non-Record Submission (Untested - Community Contribution)**
**Author:** Mikeapedia ([@mikeapedia](https://github.com/mikeapedia))
**Date:** 2026-03-31
**Status:** Tokenizer trained, corpus retokenized, pre-built data on HuggingFace. No H100 available to evaluate val_bpb.

---

## The Short Version

The competition allows custom tokenizers ("we let you bring your own"), but nobody has tried one yet. I trained a custom SentencePiece BPE tokenizer optimized for FineWeb's web-crawled text, with two key changes from the default:

1. **`split_digits=false`**: Keep number sequences as single tokens (e.g., "2024" = 1 token instead of 4)
2. **10 `user_defined_symbols`** for common web patterns (URLs, TLDs)

The tokenizer and pre-tokenized binary shards are uploaded to HuggingFace. The existing `train_gpt.py` supports custom tokenizers out of the box via `TOKENIZER_PATH` and `DATA_PATH` env vars -- no code changes needed. I'm sharing this for anyone with H100 access who wants to test it.

---

## Motivation

The default SentencePiece tokenizer treats all text equally. But FineWeb is web-crawled content, and certain patterns appear with high frequency:

- URLs (`https://www.example.com`) consume many tokens because "https", "://", "www", ".", "com" are each separate BPE merges
- Numbers like years ("2024"), prices ("$199"), and IDs get split digit-by-digit with `split_digits=true` (the default)

By pre-defining common web patterns as atomic symbols and keeping digit sequences intact, we hypothesize the tokenizer can represent web text more efficiently, potentially improving bpb.

---

## Approach

### Step 1: Corpus Frequency Analysis

Before choosing symbols, I analyzed 100,000 FineWeb documents to measure actual pattern frequencies. This was critical because **FineWeb is cleaned text, not raw HTML**. Many HTML-oriented symbols (tags, attributes) that seem obvious are actually very rare.

**Key findings from `analyze_patterns.py`:**

| Pattern | Hits per 100K docs | Status |
|---------|-------------------|--------|
| `.com` | 40,236 | Included |
| `https://` | 30,662 | Included |
| `http://` | 13,789 | Included |
| `www.` | 16,555 | Included |
| `.org` | 12,989 | Included |
| `.net` | 4,283 | Included |
| `.html` | 2,145 | Included |
| `.gov` | 1,890 | Included |
| `.edu` | 1,522 | Included |
| `.co.uk` | 978 | Included |
| `<div` | 312 | **Excluded** (too rare) |
| `href=` | 287 | **Excluded** |
| `class=` | 198 | **Excluded** |
| `<p>` | 156 | **Excluded** |

The cutoff was ~500 hits per 100K docs. Below that, the symbol wastes a BPE merge slot for negligible frequency savings.

### Step 2: Understanding the Budget

With `vocab_size=1024`:
- 3 control tokens (unk=0, bos=1, eos=2)
- 256 byte fallback tokens
- 10 user_defined_symbols
- **755 BPE merges remaining** (vs 765 with the default tokenizer)

Each user_defined_symbol costs one BPE merge slot. The 10 symbols we chose are high-frequency enough to justify that cost.

### Step 3: Tokenizer Training

```python
spm.SentencePieceTrainer.train(
    sentence_iterator=corpus_iterator,
    model_prefix="fineweb_1024_custom",
    vocab_size=1024,
    model_type="bpe",
    byte_fallback=True,
    split_digits=False,           # Key change #1
    user_defined_symbols=[        # Key change #2
        "http://", "https://", "www.",
        ".com", ".org", ".net",
        ".gov", ".html", ".edu", ".co.uk",
    ],
    max_sentence_length=16384,
    bos_id=1, eos_id=2, unk_id=0,
    num_threads=16,
    input_sentence_size=50_000_000,
    shuffle_input_sentence=True,
    train_extremely_large_corpus=True,
)
```

Trained on 5,000,000 documents from `docs_selected.jsonl` (per `manifest.json`).

### Step 4: Retokenization

The entire FineWeb corpus was retokenized into competition-format binary shards:
- **Format**: 256 x int32 header + N x uint16 LE tokens (same as original)
- **Val**: 63,770,657 tokens (1 shard)
- **Train**: 8,000,000,316 tokens (81 shards)

The custom tokenizer produces **2.8% more val tokens** than the original (63.8M vs 62.0M), meaning it's slightly less compressive overall. However, the hypothesis is that better token boundaries (keeping numbers intact, treating URLs as units) may improve model learning despite the slightly longer sequences.

---

## Pre-Built Data on HuggingFace

Everything is uploaded and ready to use:

**Repository:** [Mikeapedia/parameter-golf-data](https://huggingface.co/datasets/Mikeapedia/parameter-golf-data)

Contents:
```
tokenizers/
  fineweb_1024_custom.model     # 254KB SentencePiece model

datasets/
  fineweb10B_sp1024_custom/
    fineweb_val_000000.bin       # Validation shard
    fineweb_train_000000.bin     # Training shards (81 files)
    ...
    fineweb_train_000080.bin
```

### Download

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download tokenizer
huggingface-cli download Mikeapedia/parameter-golf-data \
  tokenizers/fineweb_1024_custom.model \
  --local-dir ./data

# Download all shards (~16GB)
huggingface-cli download Mikeapedia/parameter-golf-data \
  --include "datasets/fineweb10B_sp1024_custom/*" \
  --local-dir ./data
```

---

## How to Use

The existing `train_gpt.py` already supports custom tokenizers via environment variables. No code changes needed:

```bash
TOKENIZER_PATH=data/tokenizers/fineweb_1024_custom.model \
DATA_PATH=data/datasets/fineweb10B_sp1024_custom \
torchrun --nproc_per_node=8 train_gpt.py
```

`train_gpt.py`'s `build_sentencepiece_luts()` (line ~589) dynamically reads byte counts from any SentencePiece model file, so the BPB calculation adjusts automatically.

---

## What BPB Means with a Custom Tokenizer

BPB (bits per byte) normalizes across tokenizers:

```
BPB = (cross_entropy_loss / ln(2)) * (num_tokens / num_bytes)
```

A tokenizer that produces more tokens increases the `tokens/bytes` ratio, but if the model learns better representations from cleaner token boundaries, the `loss` term should decrease enough to compensate. The net effect on BPB is what matters -- and that's what we need H100 compute to measure.

---

## Scripts Included

- **`retokenize.py`**: End-to-end pipeline that trains the tokenizer and retokenizes the corpus into binary shards. Supports `--skip-train-tokenizer` and `--skip-retokenize` flags.
- **`analyze_patterns.py`**: Frequency analysis tool that scans FineWeb documents for web patterns and ranks them by occurrence count.

---

## Call for Testing

I don't have H100 access to test this. If you do, the data is ready to go on HuggingFace -- just download, set two env vars, and run. I'd love to know:

1. Does the custom tokenizer improve, hurt, or not affect val_bpb?
2. Does `split_digits=false` help with number-heavy validation passages?
3. Is the 2.8% token count increase a problem for training throughput?

If you run this experiment, please share results in the PR comments or Discord. Even a negative result would be valuable -- it would tell us whether tokenizer optimization is a viable axis for this competition.

---

## Reproducing the Tokenizer

If you want to retrain the tokenizer or retokenize with different settings:

```bash
# Full pipeline (requires docs_selected.jsonl from FineWeb)
python retokenize.py

# Skip tokenizer training, just retokenize with existing model
python retokenize.py --skip-train-tokenizer

# Train on fewer shards for testing
python retokenize.py --train-shards 5
```

The tokenizer training takes ~15 minutes on a modern CPU. Retokenization of 82 shards takes ~2 hours.
