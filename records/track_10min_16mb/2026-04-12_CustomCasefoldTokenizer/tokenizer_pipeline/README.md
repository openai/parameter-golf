# Tokenizer Pipeline

These scripts build the casefold v2 vocabulary from scratch. Run them in
order (each step's output feeds the next). See `CASEFOLD_TOKENIZER.md`
in the parent directory for the full technical writeup.

## Pipeline Steps

| Step | Script | What it does | Time |
|------|--------|--------------|------|
| 1 | `train_sp8192_casefold.py` | Train SentencePiece BPE on NFKC + lowercased FineWeb text | ~5 min |
| 2 | `build_clean_sp8192.py` | Remove 53 L=1 redundant tokens + 321 undertrained tokens | ~8 min |
| 3 | `casefold_candidates.py` | Merge uppercase candidate frequencies into lowercase entries | ~2 min |
| 4 | Rust `vocab-builder` | BPB-scored greedy refill of 374 freed slots (not included, see below) | ~3.2 hrs |
| 5 | `hybrid_tokenizer.py` | Convert hex vocab to SentencePiece `.model` via `create_sp_model()` | seconds |
| 6 | `swap_punct_tokens.py` | Replace 25 lowest-usage tokens with bare ASCII punctuation | seconds |
| 7 | `retokenize.py` | Retokenize full FineWeb with `--normalize casefold` | ~90 min |

## Supporting Scripts

| Script | Purpose |
|--------|---------|
| `analyze_byte_fallbacks.py` | Diagnose which tokens trigger byte fallback encoding |
| `hybrid_tokenizer.py` | Core engine shared across steps: trie, DP encoder, SP model builder |

## Dependencies

- Python 3.10+, `sentencepiece`, `unicodedata` (stdlib)
- Steps 1-3 and 5-7 are pure Python
- Step 4 requires the Rust vocab-builder (`data/vocab-builder/` in the main repo).
  Build with `cargo build --release`. It DP-encodes a 25GB corpus to score
  each candidate by marginal BPB improvement.

## Quick Start (if reproducing from scratch)

```bash
# 1. Train raw casefold BPE
uv run train_sp8192_casefold.py --vocab-size 8192

# 2. Clean (remove L=1 + undertrained)
uv run build_clean_sp8192.py \
  --model data/tokenizers/fineweb_8192_bpe_casefold_raw.model \
  --output data/tokenizers/fineweb_8192_bpe_casefold_clean.model

# 3. Merge candidate frequencies
uv run casefold_candidates.py

# 4. BPB-scored refill (Rust, ~3 hours)
data/vocab-builder/target/release/vocab-builder \
  --candidates data/tokenizers/candidates_25gb_casefold.json \
  --corpus data/tokenizers/corpus_25gb_casefold.bin \
  --base-vocab data/tokenizers/vocab_8k_casefold_clean.hex \
  --bpb-scoring --rescore-every 1 --batch-schedule "20:*" \
  --phase1-fraction 1.0 --top-k 1000000 --threads 0 \
  --output data/tokenizers/vocab_8k_casefold_refined.hex

# 5. Build SP model
uv run -c "from hybrid_tokenizer import create_sp_model; ..."

# 6. Swap low-usage tokens for bare punctuation
uv run swap_punct_tokens.py

# 7. Retokenize FineWeb
uv run retokenize.py \
  --skip-train-tokenizer \
  --tokenizer-prefix data/tokenizers/fineweb_8192_bpe_casefold_refined_v2 \
  --vocab-size 8192 \
  --output-dir data/datasets/fineweb10B_sp8192_casefold_v2 \
  --normalize casefold --sequential-val
```

Or skip all of this and use the pre-tokenized shards from HuggingFace:
`Mikeapedia/fineweb10B-sp8192-casefold-v2`
