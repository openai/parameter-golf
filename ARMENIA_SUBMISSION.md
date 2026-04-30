# Parameter Golf Armenia Submission

Entrant: Arman Karapetyan

GitHub branch for the form:

```text
https://github.com/thearmankarapetyan/parameter-golf/tree/arman-sp8192-legal-ttt-40k
```

Required files:

```text
train.py
requirements.txt
```

The root `train.py` is the submitted 8xH100 training script. It defaults to the current Armenia run configuration:

```text
VOCAB_SIZE=8192
TTT_CHUNK_TOKENS=40960
MAX_WALLCLOCK_SECONDS=590
TTT_ENABLED=1
TTT_EPOCHS=4
TTT_LR=0.005
```

FineWeb validation seeds requested by the Armenia form:

| Seed | FineWeb val_bpb |
|---:|---:|
| 1337 | TBD after 8xH100 run |
| 42 | TBD after 8xH100 run |
| 2025 | TBD after 8xH100 run |

The queued YSU 8xH100 jobs for those seeds are:

```text
32652 pg_r40_1337
32653 pg_r40_42
32654 pg_r40_2025
```

Tokenizer note for the form:

```text
Uses the default Parameter Golf SentencePiece tokenizer interface. The script loads TOKENIZER_PATH if provided, otherwise ./data/tokenizers/fineweb_8192_bpe.model with VOCAB_SIZE=8192. No custom tokenizer operations, no CaseOps, and no tokenizer-specific validation cache.
```

The 1-GPU proxy runs are not part of this submission and are not valid 8xH100 challenge logs.
