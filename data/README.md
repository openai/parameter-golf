# Data Workflows

This directory contains the dataset download helpers, tokenizer builders, and export scripts used for the challenge.

## Downloading Published Data

Download the cached FineWeb export for a tokenizer variant with:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This populates `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`.
By default it downloads the full validation split and only the first training shard, so short local runs do not need the full dataset.

To fetch more training shards, pass `--train-shards`:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 4
```

The downloader is manifest-driven and can fetch only a prefix of train shards from a larger published export. With the current shard size of `100_000_000` tokens, `10B` retokenized training tokens is `100` train shards:

```bash
MATCHED_FINEWEB_REPO_ID=your-hf-username/your-dataset-repo \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=your_50B_export_root \
python3 data/cached_challenge_fineweb.py --variant sp2048 --train-shards 100
```

Validation is always downloaded in full from the fixed `fineweb_val_*` split. Training on the first `N` train shards means training on the prefix of the same frozen shuffled export, so the data order stays aligned with the baseline for that tokenizer family.

## Exporting Data

To rebuild a larger blobstore-backed export locally or on a remote box, use:

```bash
python3 data/export_blobstore_fineweb100B_tokenizer_datasets.py \
  --output_root /tmp/matched_100B_train30Btok_even_seed1337 \
  --selection_mode even \
  --selection_seed 1337 \
  --target_train_tokens 30000000000 \
  --sp_vocab_sizes 512,1024,2048,4096 \
  --tokenizer_train_docs 5000000 \
  --skip_byte
```

This writes a shared `docs_selected.jsonl`, a `docs_selected.source_manifest.json` sidecar with source-shard metadata, tokenizers, dataset shards, and a final `manifest.json`. Copying the whole export root uploads the docs cache too, so others can retrain tokenizers or rebuild matching shards from the same selected document stream.

For the blobstore-canonical `10B` export with a fixed `50k`-doc validation prefix and byte plus `512/1024/2048` SentencePiece variants, use:

```bash
python3 data/export_blobstore_fineweb100B_tokenizer_datasets.py \
  --output_root /tmp/fineweb_blobstore100B_train10B_val50k \
  --selection_mode even \
  --selection_seed 1337 \
  --target_train_tokens 10000000000 \
  --num_val_docs 50000 \
  --sp_vocab_sizes 512,1024,2048 \
  --tokenizer_train_docs 5000000
```

This keeps the blobstore as the canonical source, takes the first `50k` documents from the blobstore val stream, then exports shuffled selected train runs until the raw GPT-2 token budget is met.

## Rebuilding Tokenizers From Published Docs

To retrain a tokenizer or re-export shards from exactly the same selected documents, first download the frozen docs cache explicitly:

```bash
MATCHED_FINEWEB_REPO_ID=your-hf-username/your-dataset-repo \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=your_50B_export_root \
python3 data/cached_challenge_fineweb.py --variant sp2048 --train-shards 0 --with-docs
```

Then run the matched exporter against the downloaded `docs_selected.jsonl`:

```bash
python3 data/export_matched_fineweb_tokenizer_datasets.py \
  --docs_jsonl ./data/docs_selected.jsonl \
  --output_root /tmp/my_custom_tokenizer_export \
  --tokenizer_config ./data/demo_tokenizer_specs.json \
  --trust_tokenizer_config_code
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
