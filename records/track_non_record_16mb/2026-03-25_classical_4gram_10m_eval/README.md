# Classical 4-gram Artifact

This is a non-record classical submission based on a discounted hashed 4-gram model exported as a compressed artifact and evaluated exactly on the full FineWeb validation split.

The model is fully non-neural:
- no transformer
- no embeddings to train
- no GPU dependence in the solver itself
- no training-data access during evaluation beyond the saved artifact

## Configuration

- Track: `non-record-16mb`
- Model: discounted hashed 4-gram with backoff to bigram and unigram
- Artifact build data: first `10,000,000` training tokens
- Artifact bytes: `14,310,783`
- Code bytes (`train_gpt.py`): `57,801`
- Total submission bytes: `14,368,584`

Command used to build the artifact:

```bash
./.venv/bin/python records/track_non_record_16mb/2026-03-25_classical_4gram_10m_eval/train_gpt.py \
  --skip-validation 1 \
  --save-state /tmp/state_ng4_10000k_comp.zlib \
  --train-pattern 'data/datasets/fineweb10B_sp1024/fineweb_train_*.bin' \
  --warmup-tokens 10000000 \
  --cache-windows '' \
  --copy-contexts '' \
  --doc-copy-contexts '' \
  --absolute-discount 0.75 \
  --ngram-contexts 3 \
  --mix-backoff-experts 0
```

Command used for the final full-validation evaluation:

```bash
./.venv/bin/python records/track_non_record_16mb/2026-03-25_classical_4gram_10m_eval/train_gpt.py \
  --max-tokens 0 \
  --report-every 5000000 \
  --load-state /tmp/state_ng4_10000k_comp.zlib \
  --cache-windows '' \
  --copy-contexts '' \
  --doc-copy-contexts '' \
  --absolute-discount 0.75 \
  --ngram-contexts 3 \
  --mix-backoff-experts 0
```

## Exact Metrics

- Full validation tokens loaded: `62,021,846`
- Predictions: `62,021,845`
- Full-validation `val_bpb`: `1.91070694`
- Full-validation wallclock: `571.97` seconds
- Validation bytes: `151,080,891`

Artifact build run:
- warmup predictions: `9,999,999`
- artifact build wallclock: `68.63` seconds

This line is much weaker than the best neural submissions, but it now satisfies the mechanical submission constraints locally:
- exact full-validation run
- artifact under `16,000,000` bytes
- single-file `train_gpt.py`
- full-validation runtime under `10` minutes on this machine

## Included Files

- `train_gpt.py` — single-file classical solver
- `submission.json` — metadata for the run
- `train.log` — exact artifact-build stdout
- `eval.log` — exact full-validation stdout
