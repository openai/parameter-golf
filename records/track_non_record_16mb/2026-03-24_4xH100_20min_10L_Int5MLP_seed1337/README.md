# Non-record: 10L Int5-MLP on 4xH100 for 1200s

This is a non-record submission based on the 10L Int5-MLP recipe, rerun on 4xH100.

It is not intended for the main 10-minute 8xH100 leaderboard. This run uses 4 GPUs with the script's built-in gradient accumulation behavior and a 1200-second wallclock cap.

## Result

- val_loss: 1.92858953
- val_bpb: 1.14222237
- bytes_total: 15803072
- stopping_early: train_time=1200141ms step=6847/20000
- final_eval_mode: sliding_window stride=64

## Command

```bash
SEED=1337 \
RUN_ID=int5_4gpu_seed1337_1200s_20260324_055951 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=1200 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

## Notes

- Final artifact used zstd compression.
- This submission is for the non-record track.
