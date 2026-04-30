# Parameter Golf Armenia Submission

Entrant: Arman Karapetyan

Submission branch:

```text
https://github.com/thearmankarapetyan/parameter-golf/tree/arman-sp8192-legal-ttt-40k
```

Required files:

```text
train.py
requirements.txt
```

Hardware:

```text
8xH100
```

Training configuration:

```text
VOCAB_SIZE=8192
ADAPT_CHUNK_TOKENS=40960
MAX_WALLCLOCK_SECONDS=590
ADAPTIVE_EVAL_ENABLED=1
ADAPT_EPOCHS=4
ADAPT_LR=0.005
TARGET_INT5_LAYERS=blocks.9.:5,blocks.10.:5
```

Data interface:

```text
Set TOKENIZER_PATH to the tokenizer model.
Set DATA_DIR to a directory containing datasets/fineweb10B_sp8192/fineweb_train_*.bin and fineweb_val_*.bin.
Alternatively set TRAIN_FILES and VAL_FILES directly to shard glob patterns.
Run with torchrun --standalone --nproc_per_node=8 train.py.
```

FineWeb validation:

| Seed | val_bpb |
|---:|---:|
| 1337 | 1.07937545 |
| 42 | 1.07898269 |
| 2025 | 1.07884878 |

Tokenizer:

```text
Uses the default Parameter Golf SentencePiece tokenizer interface.
The script loads TOKENIZER_PATH if provided.
Otherwise it uses ./data/tokenizers/fineweb_8192_bpe.model with VOCAB_SIZE=8192.
No custom tokenizer operations.
```
