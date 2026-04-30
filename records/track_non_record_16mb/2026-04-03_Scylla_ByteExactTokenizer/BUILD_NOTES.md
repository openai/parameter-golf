# Using Scylla
Scylla is a byte-exact TokenMonster-derived tokenizer path for Parameter Golf.

The packaged tokenizer artifact in this folder is `scylla.yaml`, with companion metadata `scylla.meta.npz`.

## Bundle And Runtime Requirements

Scylla depends on two pipeline requirements beyond the tokenizer artifact itself:

1. `charset:none` decoded strings must be interpreted as raw bytes via `latin-1`, not `utf-8`
2. flat binary shards need an explicit synthetic zero-byte BOS token so document boundaries survive export and exactness auditing

Any future Scylla-based dataset or eval path should preserve those requirements.

## Exactness Audit

The strict full-validation audit result is recorded in `FULL_VAL_AUDIT.json`.

Audit command used in the main repo workspace:

```bash
.venv/bin/python3 data/audit_tokenmonster_bundle.py \
  --source-root data \
  --bundle-root /Users/simon/Code/parameter-golf-local/scylla_v2_cap0_competition_export \
  --bundle-dataset fineweb10B_scylla_v2_cap0_fullbyte \
  --bundle-tokenizer tokenizers/scylla_v2_cap0_fullbyte.yaml \
  --bundle-meta tokenizers/scylla_v2_cap0_fullbyte.meta.npz \
  --strict
```

How to read those arguments:

- `--source-root`
  Root of the canonical SP1024 challenge dataset and tokenizer. In a standard repo checkout, first run:

  ```bash
  python3 data/cached_challenge_fineweb.py --variant sp1024
  ```

  This populates:

  - `data/datasets/fineweb10B_sp1024/`
  - `data/tokenizers/fineweb_1024_bpe.model`

  In that standard layout, `--source-root` is simply `data`.
- `--bundle-root`
  Root of the Scylla bundle export.
- `--bundle-dataset`
  Dataset name inside the bundle manifest. You can read this from `manifest.json` under `datasets[0].name`.
- `--bundle-tokenizer`
  Relative tokenizer artifact path inside the bundle. You can read this from `manifest.json` under `tokenizers[0].path`.
- `--bundle-meta`
  Relative metadata path inside the bundle. You can read this from `manifest.json` under `tokenizers[0].meta_path`.

If you repack or relocate Scylla, `manifest.json` is the source of truth for the last three values.

Example full-validation result:

- `source_val_docs = 50000`
- `bundle_val_docs = 50000`
- `source_bytes = 151080891`
- `meta_bytes = 151080891`
- `decoded_bytes = 151080891`
- `bad_docs = 0`
- `meta_overcount_frac = 0.0`
- `decoded_drift_frac = 0.0`

So Scylla is byte-exact on the fixed FineWeb validation text.

## Invariants For Future Scylla Work

Any future Scylla-based submission should be treated as invalid unless it preserves all of the following:

- exact validation bytes
- exact metadata denominator
- explicit document-boundary handling
- full-val equality:
- `source_bytes == meta_bytes == decoded_bytes`

## Artifact Checksums

- `scylla.yaml`
  - `sha256 = a0177241aca1871f861fec49b7f1ee737d029e8e09e320b0efd5d5ea7bee5517`
- `scylla.meta.npz`
  - `sha256 = 849652277e70b378468194b9b6d40ddc574a980522443421e1dce1016721ed72`
- `manifest.json`
  - `sha256 = 418170f7c5ccab7dcfe51e59b185f4fd6fc64c285239e635298347cd6eaff63f`
