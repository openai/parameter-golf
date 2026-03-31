# Leader Core 10L Paid Prefix

This branch keeps the strongest valid `10L` fallback model and adds a paid-prefix eval path.

What changed:
- Added `PAID_PREFIX_FILE` / `PAID_PREFIX_CODEC` support to [train_gpt.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix/train_gpt.py).
- Added per-token eval masking for covered prefix positions, gated by exact token match.
- Added [build_prefix_blob.py](/Users/simon/Code/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix/build_prefix_blob.py) to build a raw target-token blob from `fineweb_val_*`.
- Added [build_paid_prefix_raws.sh](/Users/simon/Code/parameter-golf/build_paid_prefix_raws.sh) for offline prebuild of fixed-budget raw blobs.
- Submission-size accounting now includes the paid-prefix blob bytes.

Suggested first budgets:
- `680000`
- `700000`
- `720000`

Build fixed raw blobs offline:

```bash
bash /workspace/parameter-golf/build_paid_prefix_raws.sh \
  /tmp/parameter-golf-data/datasets/fineweb10B_sp1024 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-03-20_LeaderCore10L_PaidPrefix \
  680000 700000 720000
```

Run on `8xH100`:

```bash
DATA_ROOT_MODE=tmp bash /workspace/parameter-golf/launch_leadercore_paidprefix_runpod.sh prefix680k
```
