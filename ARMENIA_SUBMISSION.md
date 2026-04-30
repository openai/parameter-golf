# Parameter Golf Armenia Submission

Entrant: Arman Karapetyan

Main files for the Armenia form:

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

```text
1337
42
2025
```

The queued YSU 8xH100 jobs for those seeds are:

```text
32652 pg_r40_1337
32653 pg_r40_42
32654 pg_r40_2025
```

Tokenizer note: this uses the default Parameter Golf SentencePiece tokenizer interface. The script loads `TOKENIZER_PATH` if provided, otherwise it uses `./data/tokenizers/fineweb_8192_bpe.model` through the standard SP8192 data layout. There are no custom tokenizer operations, no CaseOps, and no tokenizer-specific validation cache.
