# PR-1493 reproduction — save bundle for offline quantization experiments

Identical to the merged SOTA (PR-1493, 1.0810 BPB) training loop, but replaces the GPTQ + compress + eval chain with a bundle save. The bundle is the input to all quantization experiments in this axis.

## What it produces

```
bundle/
├── ema_weights.pt     # EMA-averaged state dict in float32
├── hessians.pt        # per-tensor H = X^T X from 64 calibration batches
└── template_sd.pt     # shape/dtype reference for dequantization
```

Plus a pre-quant eval (BPB ceiling, logged to stdout).

## What it does NOT do

- No GPTQ / quantization
- No compression / artifact serialization
- No post-quant eval, sliding window, TTT, or ETLB

## Run command (2×H200)

```bash
BUNDLE_DIR=./bundle \
MAX_WALLCLOCK_SECONDS=0 \
ITERATIONS=4550 \
SEED=42 \
torchrun --standalone --nproc_per_node=2 train_save_bundle.py
```

- `MAX_WALLCLOCK_SECONDS=0` disables the wallclock cap (2×H200 takes ~40 min vs 8×H100's ~10 min for the same steps).
- `ITERATIONS=4550` matches PR-1493's actual step count.
- `nproc_per_node=2` for 2×H200.

## After the run

Upload the bundle for offline iteration:
```bash
huggingface-cli upload <your-hf-repo>/parameter-golf-artifacts bundle/ pr1493_seed42/ --private
```

Then any quantization experiment loads these three files and runs its own GPTQ variant without retraining.
