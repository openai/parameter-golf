# Rerun Notes

This note records a local rerun of the latest fetched PR `#1089` head on branch `rerun`.

## Setup

- Base commit: `215193e` (`Reduce gptq_reserve_ms from 14s to 9s — reclaim ~53 training steps`)
- Executed code: `train_gpt.py` (the compressed self-extracting submission wrapper)
- Human-readable companion: `train_gpt_human.py`
- GPUs: 8x H100 80GB
- Driver: `565.57.01`
- Python: `3.12.13`
- PyTorch: `2.11.0+cu126`
- `torch.version.cuda`: `12.6`

Raw rerun log is included at:

- `logs/repro_pr1089_latest_seed42_20260330_072712.txt`

## Command

```bash
RUN_ID="repro_pr1089_latest_seed42_20260330_072712" \
DATA_PATH="/root/parameter-golf-pr1089/data/datasets/fineweb10B_sp1024" \
TOKENIZER_PATH="/root/parameter-golf-pr1089/data/tokenizers/fineweb_1024_bpe.model" \
SEED=42 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Seed 42 Comparison

Reference numbers below come from the bundled `train_seed42.log` in this record folder.

| Metric | Published `train_seed42.log` | Local rerun | Delta |
|---|---:|---:|---:|
| `step_avg` @ `step:500` | `95.04 ms` | `106.80 ms` | `+12.37%` |
| `step_avg` @ `step:1000` | `93.94 ms` | `105.78 ms` | `+12.60%` |
| `step_avg` @ `step:5000` | `93.28 ms` | `105.08 ms` | `+12.65%` |
| final `step_avg` | `93.26 ms` | `105.06 ms` | `+12.65%` |
| stop step | `6284` | `5579` | `-705` (`-11.22%`) |
| `final_int6_roundtrip_exact val_bpb` | `1.13238846` | `1.13606522` | `+0.00367676` |
| `final_int6_sliding_window_exact val_bpb` | `1.10859491` | `1.11228538` | `+0.00369047` |
| total submission size | `15,992,528` | `15,982,360` | `-10,168 bytes` |

## Main Takeaway

The main reproduction gap is throughput. The rerun is about `10-13%` slower per step than the bundled published run, and it therefore only reaches `5579` training steps inside the effective wallclock cap instead of `6284`. The worse BPB tracks that reduced step count closely.

## Important Note About The Latest PR Commit

The latest fetched PR head (`215193e`) updates `README.md` and `train_gpt_human.py`, but does **not** update the executable `train_gpt.py` wrapper.

In practice, the rerun still logged:

- `gptq:reserving 14000ms from training budget (effective cap: 586000ms)`

So the runnable submission code in `train_gpt.py` still behaves like the older compressed wrapper, even though the latest PR text says the reserve was reduced to `9000 ms`.
