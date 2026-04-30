# Parameter Golf Armenia Submission

Entrant: Arman Karapetyan

Submission branch:

```text
https://github.com/thearmankarapetyan/parameter-golf/tree/armenia-submission-final
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

FineWeb validation:

| Seed | val_bpb |
|---:|---:|
| 1337 | 1.07937545 |
| 42 | 1.07898269 |
| 2025 | 1.07884878 |

