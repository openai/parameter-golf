# V6 Dataset Modification Disclosure

This record is based on the PR1991 SP8192 Byte-PPM O=5 stack and adds one train-only micro dataset modification.

Auxiliary dataset:
- `8Planetterraforming/Parameter-Golf-V6-Privacy-Web-Filtering`

Use:
- V6 is used only as a tiny sparse train-only micro-injection.
- Injected amount: 8192 SP8192 tokens.
- Injection target: first FineWeb train shard only.
- Injection style: sparse blocks after warmup.
- V6 is not used for validation.
- V6 is not hidden eval data.

Validation:
- Official FineWeb SP8192 validation files remain untouched.
- `fineweb_val_*.bin` files are official validation files/symlinks.
- BPB is computed on FineWeb validation.

Reproduction:
- See `rebuild_and_run_v6_micro_8xh100.sh`.
- The active manifest is `v6_micro_manifest.json`.

Review note:
Because this record modifies training data, it should be reviewed more carefully than a pure FineWeb run. The intended legality boundary is train-only V6 micro-injection with untouched FineWeb validation.
