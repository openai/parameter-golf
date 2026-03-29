# Data Workflows

This directory contains the dataset download helpers and export scripts used for the challenge.

Canonical local layout:
- `data/datasets/<dataset_name>/`
- `data/tokenizers/`
- `data/manifest.json`
- `data/docs_selected.jsonl`
- `data/docs_selected.source_manifest.json`

## Downloading Published Data

Download the cached FineWeb export for a tokenizer variant with:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This populates `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`.
By default it downloads the full validation split and 8B training tokens (80 train shards).

To fetch more training shards, pass `--train-shards`:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 180
```

The downloader is manifest-driven and can fetch only a prefix of train shards from a larger published export. With the current shard size of `100_000_000` tokens, `10B` retokenized training tokens is `100` train shards:

```bash
MATCHED_FINEWEB_REPO_ID=your-hf-username/your-dataset-repo \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=your_50B_export_root \
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 100
```

Validation is always downloaded in full from the fixed `fineweb_val_*` split. Training on the first `N` train shards means training on the prefix of the same frozen shuffled export, so the data order stays aligned with the baseline for that tokenizer family.

The default published repo is `willdepueoai/parameter-golf`, with the export rooted under the repo subdirectory `datasets/`.

### Flat Byte-Shard Repos

Some byte-level repos publish flat shard files at the repo root instead of a manifest-driven export tree. For example, `mistobaan/fineweb10B_bytes` appears to publish `fineweb_train_*.bin` shards directly.

Use `--flat-repo` for that case:

```bash
python3 data/cached_challenge_fineweb.py \
  --repo-id mistobaan/fineweb10B_bytes \
  --remote-root '' \
  --flat-repo \
  --variant bytes \
  --train-shards 80 \
  --val-from-local /path/to/fineweb10B_bytes
```

Notes:
- `--variant bytes` maps to the local dataset directory `data/datasets/fineweb10B_bytes/`.
- `--remote-root ''` means the `.bin` shards live at the repo root rather than under `datasets/...`.
- Many flat byte repos publish training shards only. In that case, supply validation with `--val-from-local` from another compatible byte export, or use `--val-repo-id` and `--val-remote-root` to fetch validation from a separate repo/path.

## Rebuilding Tokenizers From Published Docs

To retrain a tokenizer or re-export shards from exactly the same selected documents, run the standalone retokenizer against the published docs cache:

```bash
python3 data/download_hf_docs_and_tokenize.py \
  --repo-id your-hf-username/your-dataset-repo \
  --remote-root your_50B_export_root \
  --output-root /tmp/my_custom_tokenizer_export \
  --tokenizer-config ./data/tokenizer_specs.json
```

The sidecar `docs_selected.source_manifest.json` includes `docs_sha256`, so users can verify they are rebuilding from the exact same document list and order as the baseline export.

## Useful Knobs

For CPU-heavy exports, useful knobs are:

```bash
MATCHED_FINEWEB_SP_BATCH_SIZE=2048
MATCHED_FINEWEB_TOKENIZER_THREADS=16
MATCHED_FINEWEB_TIKTOKEN_THREADS=16
MATCHED_FINEWEB_GPT2_DECODE_BATCH_SIZE=512
```

These control batched tokenizer encoding during shard export, tokenizer thread count, tiktoken thread count, and batched GPT-2 decode for the blobstore docs-cache path.
