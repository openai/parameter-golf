# Non-Record Submission: Single-GPU Blackwell Port of 10L Int5-MLP + BigramHash(10240)

This is a non-record unlimited-compute submission that ports the merged `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` recipe to a single `RTX PRO 6000 Blackwell Server Edition`.

The goal was not to challenge the live 8xH100 frontier under the 10-minute rules. The goal was to make the recipe run cleanly on one widely available GPU while keeping the artifact under 16MB and preserving most of the original architecture: 10 layers, `3x` MLP, SmearGate, BigramHash(10240), mixed low-precision export, and late SWA.

## Best Completed Run

Run ID: `rtx6000_45m_b131k_s64`

- post-quant exact `val_bpb`: **1.19349046**
- post-quant exact `val_loss`: **2.01515331**
- pre-quant `val_bpb`: `1.2082`
- pre-quant `val_loss`: `2.0399`
- steps completed: `14085`
- wallclock: `2700s`
- total artifact size: `15,691,796` bytes
- GPU: `1x RTX PRO 6000 Blackwell Server Edition`

## What Changed From The Merged 10L Record

1. Portable AMP dtype selection: `bf16` on newer CUDA GPUs, `fp16` fallback on older GPUs.
2. SDPA backend probing plus a manual KV expansion fallback when native `enable_gqa=True` support is unavailable.
3. Optional `LOAD_MODEL_PATH` restore before `torch.compile()` to support eval-only reloads.
4. Single-GPU runtime tuning through env vars: smaller batch (`131072` tokens), longer wallclock, and controllable sliding-window eval.

These changes were enough to run the recipe on a single Blackwell GPU without changing the underlying model family.

## Best Run Command

```bash
RUN_ID=rtx6000_45m_b131k_s64 \
DATA_PATH=/home/zeus/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/home/zeus/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
TRAIN_BATCH_TOKENS=131072 \
MAX_WALLCLOCK_SECONDS=2700 \
WARMUP_STEPS=5 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=64 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Follow-up Run

A longer follow-up run, `rtx6000_65m_b131k_s32_eb512`, trained to step `20330` with pre-quant `val_bpb 1.2011`, but its sliding-window evaluation was interrupted after the Lightning cloudspace was reclaimed. It is included in `results.tsv` and the preserved log tail for completeness, but it is not used as the submission score.

## Included Files

- `train_gpt.py`: single-GPU portable code snapshot used for the Blackwell runs
- `results.tsv`: exact submission result plus one incomplete follow-up run
- `train_rtx6000_45m_b131k_s64_tail.log`: preserved tail of the Lightning-produced best-run log
- `train_rtx6000_65m_b131k_s32_eb512_tail.log`: preserved tail of the longer follow-up run
- `submission.json`: leaderboard metadata
