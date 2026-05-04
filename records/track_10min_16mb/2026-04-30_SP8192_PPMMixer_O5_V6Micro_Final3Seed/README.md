# SP8192 Byte-PPM O=5 + V6 Privacy-Web-Filtering Micro Final

Base:
- PR1991 SP8192 Byte-PPM Mixer O=5.

Modification:
- A tiny V6 train-only sparse micro-injection is applied to the first FineWeb train shard.
- FineWeb validation files are untouched official symlinks.
- V6 is not used as validation or evaluation data.
- No validation leakage is intended.

Risk:
- This is a final-time ablation. If it underperforms pure PR1991, pure PR1991 remains the safer baseline.

## V6 Dataset Modification Disclosure

This run uses a tiny train-only V6 sparse micro-injection.  
See `V6_DATASET_DISCLOSURE.md`.

Important:
- FineWeb validation files are untouched official `fineweb_val_*.bin`.
- V6 is not validation data.
- V6 is not hidden eval data.
- `rebuild_and_run_v6_micro_8xh100.sh` documents the rebuild and run procedure.
