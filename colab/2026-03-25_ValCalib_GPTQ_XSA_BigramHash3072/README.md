# Colab Replica: 2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072

This folder provides a Google Colab entrypoint for the record training run in [records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py).

The goal is to keep the training stack as close as possible to the original record while making only the changes needed for a single Colab GPU.

## What stays the same

- The actual trainer logic is still the original record script.
- Model architecture, optimizer split, EMA/SWA behavior, AR self-generated GPTQ calibration, full-Hessian GPTQ, selective pruning, and LZMA compression are unchanged.
- The key run settings remain the same by default:
  - `BIGRAM_VOCAB_SIZE=3072`
  - `BIGRAM_DIM=112`
  - `WARMDOWN_ITERS=4000`
  - `TARGET_MB=15.9`
  - `TRAIN_SEQ_LEN=2048`
  - `EVAL_SEQ_LEN=2048`

## What changes for Colab

- Training is forced onto the first 10 FineWeb training shards by creating a local 10-shard data view.
- The default train batch is reduced to `65536` tokens for single-GPU memory limits.
- `flash_attn_interface.py` provides a compatibility shim backed by PyTorch SDPA, so the original script can run without FlashAttention 3 or Hopper-only wheels.
- `sitecustomize.py` adds runtime-only portability toggles:
  - disable `torch.compile`
  - disable fused Adam if the Colab torch build needs it
  - remap bf16 calls to fp16 on GPUs without CUDA bf16 support

These are runtime shims only. The record script itself is not edited.

## Tradeoffs

This folder is a practical Colab reproduction path, not a hardware-faithful replay of the original run.

- The original record was trained on 8 H100 GPUs.
- This Colab version is meant for 1 GPU.
- Because of that, the script has to use a much smaller batch.
- You asked for 10 train shards, so this version also trains on less data than the original full setup.
- The original run used FlashAttention 3 on Hopper GPUs. Colab may not provide that exact hardware or wheel support, so this version uses PyTorch's built-in attention path when needed.

What that means in practice:

- Training will be slower.
- Memory usage will be lower.
- Final validation numbers will likely be worse than the record run.
- The architecture and quantization recipe are still kept as close as possible to the original.

## `bf16` and `fp16`, in plain language

These are two 16-bit number formats used to make GPU training faster and smaller than regular 32-bit math.

- `bf16` means `bfloat16`.
- `fp16` means `float16`.

Both use 16 bits, but they behave differently:

- `bf16` is usually more numerically stable for training.
- `fp16` is older and more widely supported.
- Some Colab GPUs support `bf16` well.
- Some older Colab GPUs do not, so the script falls back to `fp16`.

You do not need to set this manually in the normal case. `run.sh` checks the GPU and decides:

- if the GPU supports `bf16`, it keeps the original-style `bf16` path
- if the GPU does not support `bf16`, it automatically switches to `fp16`

Why this matters:

- `bf16` is closer to what the original training run likely wants.
- `fp16` is a compatibility fallback so the script still runs on more Colab GPU types.
- If `fp16` is used, training can be a bit less stable and results can differ more from the original run.

## Files

- [train_gpt.py](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py): thin local entrypoint that runs the original record script.
- [run.sh](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/run.sh): prepares the 10-shard view, sets env vars, and launches training.
- [flash_attn_interface.py](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/flash_attn_interface.py): FlashAttention 3 API shim implemented with SDPA.
- [sitecustomize.py](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/sitecustomize.py): Colab compatibility patch layer.
- [requirements.txt](/Users/ian_muliterno/Documents/GitHub/parameter-golf-fork/colab/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/requirements.txt): extra Python deps for this folder.

## Colab usage

Run Colab with a GPU runtime. A100 or L4 is the closest match. T4 should also launch, but it will use the fp16 fallback and will be slower.

```bash
git clone https://github.com/IanniMuliterno/parameter-golf.git
cd parameter-golf/colab/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072
python3 -m pip install -r requirements.txt
bash run.sh
```

If you want `run.sh` to install the extra Python dependencies itself:

```bash
INSTALL_DEPS=1 bash run.sh
```

## Useful knobs

Override env vars inline when needed:

```bash
TRAIN_BATCH_TOKENS=32768 SEED=42 bash run.sh
```

```bash
PG_COLAB_DISABLE_COMPILE=0 PG_COLAB_DISABLE_FUSED_ADAM=1 bash run.sh
```

## Outputs

Run from this folder and the original trainer will emit its outputs here:

- `logs/`
- `final_model.pt`
- `final_model.int6.ptz`
- `runtime_data/fineweb10B_sp1024_10shards/`

## If you want fewer automatic fallbacks

By default, the launcher tries to be practical and forgiving on Colab.

- `PG_COLAB_DISABLE_COMPILE=1` keeps `torch.compile` off by default because Colab environments can be inconsistent.
- `PG_COLAB_DISABLE_FUSED_ADAM=0` leaves fused Adam on unless you explicitly disable it.
- `PG_COLAB_FORCE_FP16` is chosen automatically based on the GPU.

Examples:

```bash
PG_COLAB_DISABLE_COMPILE=0 bash run.sh
```

```bash
PG_COLAB_DISABLE_FUSED_ADAM=1 bash run.sh
```
