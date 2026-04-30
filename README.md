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
TTT_CHUNK_TOKENS=40960
MAX_WALLCLOCK_SECONDS=590
TTT_ENABLED=1
TTT_EPOCHS=4
TTT_LR=0.005
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
