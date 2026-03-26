# Classical Doc-Copy 16.3M LZMA Artifact

Non-record classical submission using a document-local copy expert over a discounted hashed 4-gram backoff chain, with packed n-gram token storage and `lzma` state compression.

Model summary:
- train warmup: first `16,300,000` tokens from the official `fineweb10B_sp1024` train export
- saved artifact: `15,638,788` bytes
- code bytes (`train_gpt.py`): `66,221`
- total bytes: `15,705,009`
- evaluation: official `fineweb_val_*` split, artifact-only, online adaptation only after each scored token
- local verification hardware: `Apple M4` CPU

Exact full-validation result:
- `val_bpb=1.81114207`
- validation tokens: `62,021,846`
- predictions: `62,021,845`
- evaluation wallclock: `280.19s`

Build command:

```bash
python records/track_non_record_16mb/2026-03-27_classical_doccopy_16p3m_lzma_eval/train_gpt.py \
  --train-pattern 'data/datasets/fineweb10B_sp1024/fineweb_train_*.bin' \
  --save-state /tmp/state_submit_doccopy_16p3m_packaged.xz \
  --state-compression lzma \
  --warmup-tokens 16300000 \
  --ngram-contexts 3 \
  --cache-windows '' \
  --copy-contexts '' \
  --doc-cache-windows '' \
  --doc-copy-contexts 2 \
  --absolute-discount 0.75 \
  --continuation-unigram 1 \
  --unigram-alpha 0.5 \
  --mix-backoff-experts 0 \
  --eta 0.3 \
  --share 0.05 \
  --skip-validation 1
```

Eval command:

```bash
python records/track_non_record_16mb/2026-03-27_classical_doccopy_16p3m_lzma_eval/train_gpt.py \
  --load-state /tmp/state_submit_doccopy_16p3m_packaged.xz \
  --warmup-tokens 0 \
  --ngram-contexts 3 \
  --cache-windows '' \
  --copy-contexts '' \
  --doc-cache-windows '' \
  --doc-copy-contexts 2 \
  --absolute-discount 0.85 \
  --continuation-unigram 1 \
  --unigram-alpha 0.5 \
  --mix-backoff-experts 0 \
  --eta 0 \
  --share 0 \
  --expert-weights 0,1 \
  --copy-alpha 0.75 \
  --copy-decay-power 0.3 \
  --copy-window 200000 \
  --copy-max-matches 32 \
  --copy-store-limit 32 \
  --max-tokens 0 \
  --report-every 5000000
```

Notes:
- the active scoring path is effectively `doc_copy_ctx2` only, with the discounted `ngram_4 -> bigram -> unigram` chain used as backoff
- packed `10`-bit follower token storage and `lzma` state compression are what made the `16.3M` artifact fit under the `16MB` cap
- no validation-time access to training shards is required or used
- the artifact is rebuilt by `train_gpt.py`; it is not checked into the repo
