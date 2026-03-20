# Leader Core 10L Paid Prefix

This branch keeps the strongest valid `10L` fallback model and adds a paid-prefix eval path.

What changed:
- Added `PAID_PREFIX_FILE` / `PAID_PREFIX_CODEC` support to [train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix/train_gpt.py).
- Added per-token eval masking for covered prefix positions, gated by exact token match.
- Added [build_prefix_blob.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix/build_prefix_blob.py) to build a compressed target-token blob from `fineweb_val_*`.
- Submission-size accounting now includes the paid-prefix blob bytes.

Suggested first budgets:
- `512000`
- `768000`
- `1048576`

Build a blob:

```bash
python3 /workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix/build_prefix_blob.py \
  --val-dir /tmp/parameter-golf-data/datasets/fineweb10B_sp1024 \
  --output /workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix/prefix_768k.xz \
  --budget-bytes 768000
```

Run on `8xH100`:

```bash
DATA_ROOT_MODE=tmp bash /workspace/parameter-golf/launch_leadercore_paidprefix_runpod.sh prefix768k
```
