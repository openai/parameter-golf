# Parameter Golf Armenia Submission

Name: Arman Karapetyan

GitHub:

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

FineWeb validation:

| Seed | val_bpb |
|---:|---:|
| 1337 | 1.07937545 |
| 42 | 1.07898269 |
| 2025 | 1.07884878 |

Tokenizer note:

```text
Uses the default Parameter Golf SentencePiece tokenizer interface. The script loads TOKENIZER_PATH if provided, otherwise ./data/tokenizers/fineweb_8192_bpe.model with VOCAB_SIZE=8192. No custom tokenizer operations.
```
