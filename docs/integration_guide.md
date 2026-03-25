# Integration Guide: Plugging BESE into Parameter Golf

## Overview

Three things need to change to use BESE+BPE in the Parameter Golf challenge:

1. **Data pipeline** - Re-tokenize FineWeb with BESE+BPE
2. **Training script** - Load BESE+BPE instead of SentencePiece
3. **Model config** - Set VOCAB_SIZE to match your tokenizer

Everything else (model architecture, optimizer, quantization, evaluation loop) stays identical.

---

## Step 1: Re-tokenize FineWeb

### Option A: Modify tokenizer_specs.json

Add your tokenizer to the config:

```json
{
  "tokenizers": [
    {
      "name": "bese_bpe_250",
      "dataset_suffix": "bese250",
      "builder": "build_bese_bpe_tokenizer",
      "num_merges": 250,
      "tokenizer_train_docs": 100000
    }
  ]
}
```

Then register `build_bese_bpe_tokenizer` in `download_hf_docs_and_tokenize.py`:

```python
from bese_bpe_tokenizer import build_bese_bpe_tokenizer

# In the tokenizer_kind() function, add:
if builder_name == "build_bese_bpe_tokenizer":
    return "bese_bpe"

# In the main export loop, add the builder
```

### Option B: Standalone re-export script

Implemented as [`scripts/export_shards.py`](../scripts/export_shards.py). It:
1. Loads JSONL with a `text` field (e.g. `docs_selected.jsonl`)
2. Loads a trained `BESEBPETokenizer` from JSON
3. Encodes documents and writes `fineweb_train_*.bin` / `fineweb_val_*.bin` shards

Train merges first, e.g.:

```bash
python scripts/train_bpe_jsonl.py --input data/docs_selected.jsonl \
  --output data/tokenizers/bese_bpe_250.json --num-merges 250 --max-docs 100000
```

The shard format is simple:
- 256-int32 header (magic=20240520, version=1, token_count=N, rest zeros)
- N uint16 token values

---

## Step 2: Modify train_gpt.py (or use the fork)

A ready-made fork lives at [`integration/train_gpt_bese.py`](../integration/train_gpt_bese.py): if `TOKENIZER_PATH` ends with `.json`, it loads `BESEBPETokenizer`; otherwise it uses SentencePiece `.model` as upstream.

### Replace SentencePiece loading (manual patch)

Current code (line ~805):
```python
if not args.tokenizer_path.endswith(".model"):
    raise ValueError(...)
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
```

Replace with:
```python
from bese_bpe_tokenizer import BESEBPETokenizer

tok = BESEBPETokenizer.load(args.tokenizer_path)
assert tok.vocab_size == args.vocab_size
```

### Replace build_sentencepiece_luts

Current code (line ~815):
```python
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
    sp, args.vocab_size, device
)
```

Replace with:
```python
base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tok.build_luts_for_training(device)
```

The `build_luts_for_training()` method returns the exact same tuple format, so eval_val() works without changes.

### Set environment variables

```bash
VOCAB_SIZE=288 \
DATA_PATH=./data/datasets/fineweb10B_bese250/ \
TOKENIZER_PATH=./data/tokenizers/bese_bpe.json \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## Step 3: Verify BPB correctness

Before submitting, verify that the total byte count from your tokenizer matches the original text:

```python
tok = BESEBPETokenizer.load("bese_bpe.json")
bpt = tok.get_bytes_per_token_lut()

# For each validation document:
encoded = tok.encode(text)
token_bytes = sum(bpt[t] for t in encoded)
utf8_bytes = len(text.encode('utf-8'))
assert token_bytes == utf8_bytes, f"Byte mismatch: {token_bytes} vs {utf8_bytes}"
```

This check is critical. OpenAI will scrutinize tokenizer modifications for BPB correctness.

---

## Step 4: Tune the number of BPE merges

The optimal number of merges depends on the tradeoff between:
- **More merges** = shorter sequences (better for attention) but larger embedding table
- **Fewer merges** = smaller embedding table but longer sequences

Test on a single GPU with different merge counts (100, 200, 300, 400) and compare:
- Sequence length on validation set
- Training steps per 10 minutes
- Final BPB

---

## Combining with the standard competition stack

The tokenizer change is orthogonal to architecture improvements. You can layer on top:

- XSA (partial, on last 4 layers)
- EMA weight averaging
- SmearGate + BigramHash + OrthoInit
- Muon optimizer with WD=0.04
- FlashAttention 3
- Int6 quantization with Late QAT
- Sliding window evaluation

The key advantage: your smaller embedding table lets you fit more transformer layers,
and competition data shows more layers consistently wins.
