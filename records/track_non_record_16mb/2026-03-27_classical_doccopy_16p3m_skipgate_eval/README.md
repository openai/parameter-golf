# Classical Doc-Copy 16.3M Skip-Gate Artifact

Non-record classical submission using a discounted hashed 4-gram backoff chain with document-local copy and skip-copy experts. The active eval path uses a fixed three-expert blend plus an instantaneous confidence gate.

Model summary:
- train warmup: first `16,300,000` tokens from the official `fineweb10B_sp1024` train export
- saved artifact: `15,645,820` bytes
- code bytes: `51,122`
- total bytes: `15,696,942`
- evaluation: official `fineweb_val_*` split, artifact-only, online adaptation only after each scored token
- local verification hardware: `Apple M4` CPU

Exact full-validation result:
- `val_bpb=1.79538936`
- validation tokens: `62,021,846`
- predictions: `62,021,845`
- evaluation wallclock: `471.76s`

Build command:

```bash
python records/track_non_record_16mb/2026-03-27_classical_doccopy_16p3m_skipgate_eval/train_gpt.py \
  --train-pattern 'data/datasets/fineweb10B_sp1024/fineweb_train_*.bin' \
  --save-state /tmp/state_submit_doccopy_16p3m_skipgate_packaged.xz \
  --state-compression lzma \
  --warmup-tokens 16300000 \
  --ngram-contexts 3 \
  --doc-copy-contexts 2 \
  --doc-skip-copy-contexts 1-3,1-4 \
  --absolute-discount 0.75 \
  --continuation-unigram 1 \
  --unigram-alpha 0.5 \
  --eta 0.3 \
  --share 0.05 \
  --skip-validation 1
```

Eval command:

```bash
python records/track_non_record_16mb/2026-03-27_classical_doccopy_16p3m_skipgate_eval/train_gpt.py \
  --load-state /tmp/state_submit_doccopy_16p3m_skipgate_packaged.xz \
  --warmup-tokens 0 \
  --ngram-contexts 3 \
  --doc-copy-contexts 2 \
  --doc-skip-copy-contexts 1-3,1-4 \
  --absolute-discount 0.85 \
  --continuation-unigram 1 \
  --unigram-alpha 0.5 \
  --eta 0 \
  --share 0 \
  --instantaneous-eta 1.0 \
  --expert-weights 0.00,0.78,0.14,0.08 \
  --copy-alpha 0.75 \
  --copy-decay-power 0.3 \
  --copy-window 200000 \
  --copy-max-matches 32 \
  --copy-store-limit 32 \
  --max-tokens 0 \
  --report-every 5000000
```

Notes:
- the active scoring path is an instantaneous-gated fixed blend over `doc_copy_ctx2`, `doc_skip_copy_1-3`, and `doc_skip_copy_1-4`
- packed `10`-bit follower token storage and `lzma` state compression are what made the `16.3M` artifact fit under the `16MB` cap
- no validation-time access to training shards is required or used
- the artifact is rebuilt by `train_gpt.py`; it is not checked into the repo
