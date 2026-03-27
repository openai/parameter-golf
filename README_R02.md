# R02: Lower LR + Warmdown + Sliding Eval

This README captures the current `R02` code/config snapshot in the style of the `/records` READMEs.

This is not a submission record yet. The numbers below come from local MLX validation before a CUDA run.

## Summary

Settings-only sweep on top of the existing baseline:

- `2026-03-18_LowerLR`: lower default learning rates
- `2026-03-19_WarmdownQuantization`: `WARMDOWN_ITERS=20000`
- sliding-window final eval (`EVAL_STRIDE=64`)

No architecture changes in this step.

## Trainer Changes In This Snapshot

- `train_gpt.py`
  - default LR / warmdown updated to the R02 values
  - final int8 roundtrip eval now always uses sliding-window scoring
  - compiled `forward_logits` path added for eval
- `train_gpt_mlx.py`
  - same LR / warmdown defaults for local validation
  - eval-only checkpoint loading for same-checkpoint A/B
- `mlx_local.sh`
  - unified local `setup`, `download`, `run`, and `compare` helper

## Configuration

- `WARMDOWN_ITERS=20000` (was `1200`)
- `MATRIX_LR=0.02` (was `0.04`)
- `SCALAR_LR=0.02` (was `0.04`)
- `TIED_EMBED_LR=0.03` (was `0.05`)
- `EVAL_STRIDE=64`
- `EVAL_BATCH_SEQS=32`

## Local Validation

Command:

```bash
ITERATIONS=2000 TRAIN_BATCH_TOKENS=32768 VAL_BATCH_SIZE=32768 bash mlx_local.sh run sliding
```

Key metrics from `logs/mlx_sliding_20260327_094211.txt`:

- Timed training stopped at `465/2000` steps due to the `600s` wallclock cap
- Pre-quant eval at stop: `val_loss:3.1569`, `val_bpb:1.8697`
- Post-quant roundtrip eval: `val_loss:3.16510128`, `val_bpb:1.87455366`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.87455366`
- Serialized model int8+zlib: `8304641 bytes`

Compared against the earlier local sliding run (`logs/mlx_sliding_20260326_201929.txt`):

- Previous post-quant roundtrip eval: `val_bpb:2.40618978`
- New post-quant roundtrip eval: `val_bpb:1.87455366`
- Delta: `-0.53163612 BPB`
- Relative improvement: `-22.1%`
- Previous int8+zlib artifact: `11260463 bytes`
- New int8+zlib artifact: `8304641 bytes`
- Artifact delta: `-2955822 bytes` (`-26.2%`)

This is not a clean settings-only A/B because the newer run also trained longer than the earlier smoke run. It should be read as overall local progress.

## Same-Checkpoint Eval Comparison

Same `.ptz` checkpoint, evaluated two ways:

- Standard eval: `val_bpb:2.40675099`
- Sliding eval: `val_bpb:2.40618978`
- Sliding eval delta: `-0.00056121 BPB`
- Eval time ratio on MLX: about `15.2x` slower

This confirms that sliding evaluation itself is positive on the same checkpoint, even though most of the gain appears to come from the LR / warmdown change.

## How To Run

Local MLX setup:

```bash
bash mlx_local.sh setup
bash mlx_local.sh download 10
```

Local MLX run:

```bash
bash mlx_local.sh run sliding
```

Same-checkpoint compare:

```bash
bash mlx_local.sh compare logs/<model>.int8.ptz
```

CUDA run with current defaults:

```bash
RUN_ID=r02_cuda_8gpu \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
