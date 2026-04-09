# Data Lineage Verification

SP4096 data from [sproos/parameter-golf-tokenizers](https://huggingface.co/sproos/parameter-golf-tokenizers) is tokenized from the **same FineWeb documents** as the official SP1024 data in [willdepueoai/parameter-golf](https://huggingface.co/datasets/willdepueoai/parameter-golf).

## Cryptographic Hash Match

`docs_selected.source_manifest.json` is **byte-for-byte identical** in both repos:

| Field | Official (willdepueoai) | Sproos |
|-------|------------------------|--------|
| `docs_sha256` | `84386dfa7b339a...d19bc7` | `84386dfa7b339a...d19bc7` |
| `num_docs` | 15,368,808 | 15,368,808 |
| `docs_val` | 50,000 | 50,000 |
| `docs_train` | 15,318,808 | 15,318,808 |
| `docs_bytes` | 48,166,275,520 | 48,166,275,520 |
| `selection_seed` | 1337 | 1337 |

## Val Token Counts

Same 50,000 documents, different tokenizations:

| Tokenizer | Val Tokens | Val Bytes |
|-----------|-----------|-----------|
| Official SP1024 | 62,021,846 | ~151M |
| Sproos SP4096 | 44,847,738 | ~151M |

Byte count is identical (same UTF-8 text). Token count differs because SP4096 has larger vocabulary → fewer tokens per byte.

## Lineage Chain

1. Official FineWeb 50k eval docs selected with `selection_seed=1337`
2. Documents hashed: `docs_sha256 = 84386dfa7b339a...d19bc7`
3. Sproos retokenized the same documents with SP4096 BPE
4. Sproos's manifest references `remote_repo_id = "willdepueoai/parameter-golf"`
