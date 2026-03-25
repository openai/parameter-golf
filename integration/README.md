# Parameter Golf integration (BESE+BPE)

## `train_gpt_bese.py`

Fork of upstream [`train_gpt.py`](https://github.com/openai/parameter-golf/blob/main/train_gpt.py) with optional **BESE+BPE** tokenizer support.

- **SentencePiece (default):** `TOKENIZER_PATH` ends with `.model` — unchanged behavior.
- **BESE+BPE:** `TOKENIZER_PATH` points to a `.json` file saved by `BESEBPETokenizer.save()` (see `tokenizer/bese_bpe_tokenizer.py`).

### Path resolution

The script adds `BESE_TOKENIZER_ROOT` (default: `../tokenizer` relative to this file) to `sys.path` so `bese_bpe_tokenizer` imports work when this repo layout is preserved.

If you copy only `integration/train_gpt_bese.py` into the upstream repo, either:

- Copy the `tokenizer/` package next to it, or  
- `export BESE_TOKENIZER_ROOT=/path/to/parameter-golf-bese/tokenizer`

### Example run

```bash
cd /path/to/parameter-golf-bese
export VOCAB_SIZE=288   # 38 + 250 merges — set to your tokenizer’s vocab_size
export DATA_PATH=./data/datasets/my_bese_dataset/
export TOKENIZER_PATH=./data/tokenizers/bese_bpe_250.json
torchrun --standalone --nproc_per_node=1 integration/train_gpt_bese.py
```

Use `NUM_LAYERS`, `MODEL_DIM`, etc. as in upstream. Shards must be produced with the **same** tokenizer as `TOKENIZER_PATH` (see `scripts/export_shards.py`).
