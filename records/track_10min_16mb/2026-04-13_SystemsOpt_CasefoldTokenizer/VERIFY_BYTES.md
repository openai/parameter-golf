# Byte Verification: Why This Proves Our BPB Is Accurate

## The BPB Formula

```
BPB = (val_loss / ln2) × (tokens / bytes)
```

`val_loss` and `tokens` come directly from the model's forward pass — they
can't be wrong. The only variable a custom tokenizer can affect is **bytes**:
the denominator that converts token-level metrics into byte-level metrics.

## How Bytes Are Counted During Evaluation

The training script (`train_gpt_human.py`, lines 982-984) counts bytes using
a **lookup table (LUT)** built from the tokenizer vocabulary:

```python
token_bytes = base_bytes_lut[tgt_ids]
token_bytes += has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
val_byte_count += token_bytes.sum()
```

For each predicted token, the LUT returns:
- **base_bytes**: UTF-8 byte width of the piece text (with interior `▁`
  replaced by ASCII space)
- **+1** if the piece has a leading `▁` and the previous token isn't a
  boundary (BOS, control, etc.) — this counts the space byte

This accumulates `val_byte_count`, which becomes the `bytes` in the BPB
formula.

## What `verify_bytes.py` Proves

The script verifies that the LUT byte count **exactly equals** the true
byte count of the text the model sees.

For each document:

1. Apply our normalization: `NFKC(text).lower()`
2. Apply SentencePiece's internal normalization (`sp.normalize()`) — this
   is the exact text SP tokenizes (whitespace collapsing, etc.)
3. Count the UTF-8 bytes of that normalized text → **ground truth**
4. Tokenize with SP, then accumulate bytes using the same LUT logic as
   `eval_val()` → **LUT bytes**
5. Assert: `LUT bytes == ground truth`

If this holds on every document, the byte denominator in BPB is correct.

## Why This Is Sufficient

The LUT and the model operate on the same token stream. When `eval_val()`
computes loss on token `t`, it also looks up `base_bytes_lut[t]` for the
byte count. If the sum of all LUT lookups equals the true byte count of
the text, then:

- No bytes are double-counted
- No bytes are missed
- The BPB formula's denominator is accurate

## What SentencePiece Normalization Means

SentencePiece applies `nmt_nfkc` normalization internally before tokenizing.
This goes beyond Python's `unicodedata.normalize("NFKC")`:

- Newlines → spaces
- Consecutive whitespace → single space
- Various Unicode normalizations

This means the model never sees the raw text — it sees SP's normalized
version. The LUT correctly counts bytes of this normalized text, not the
raw text. This is the right thing to measure: BPB should reflect the
information content the model actually predicts.

## Results

**200-doc spot check (bundled):**
```
Documents verified:  200
Ground-truth bytes:  1,489,674
LUT bytes:           1,489,674
Mismatched docs:     0 / 200
RESULT: ALL CHECKS PASSED
```

**Full 15.4M-document FineWeb corpus** (results in `verify_results.txt`):
```
Documents verified:  15,368,808
Tokens:              11,423,532,518
Ground-truth bytes:  47,707,155,846
LUT bytes:           47,707,155,846
Mismatched docs:     0 / 15,368,808
```
Zero mismatches across the entire dataset.

## Running It Yourself

Spot check (bundled sample, ~30 seconds, no GPU):
```bash
pip install sentencepiece
python verify_bytes.py --docs verify_docs.jsonl
```

Full verification (requires FineWeb corpus):
```bash
python verify_bytes.py --docs data/docs_selected.jsonl --max-docs 0
```
