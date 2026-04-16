# PR-1493 reproduction — save bundle for offline quantization experiments

Identical to the merged SOTA (PR-1493, 1.0810 BPB) training loop, but replaces the GPTQ + compress + post-quant eval chain with a bundle save. The bundle is the input to all quantization experiments in this axis.

## Files

- `train_save_bundle.py` — patched PR-1493 training script. Full training + EMA + pre-quant eval, then saves bundle instead of quantizing.
- `quantize_bundle.py` — loads a saved bundle, applies PR-1493 GPTQ (or any variant via env vars), runs eval (pre-quant + post-quant, standard + sliding window). No TTT. This is the template for Q1-Q11 experiments.
- `modal_launcher.py` — Modal runner. Supports `--mode prefetch / train / quantize`. Train uses 8×H100; quantize uses 1×H100.
- `README.md` (this file).

## What the bundle contains

```
<BUNDLE_DIR>/
├── ema_weights.pt     # EMA-averaged state dict (float32)
├── hessians.pt        # per-tensor H = X^T X from 64 calibration batches
└── template_sd.pt     # shape/dtype reference for dequantization
```

Plus `pre-quantization post-ema val_bpb: ...` logged to stdout — the ceiling BPB before any quantization.

## What it does NOT do

- No GPTQ / quantization
- No compression / artifact serialization
- No post-quant eval, sliding window, TTT, or ETLB

## Launch on Modal (8×H100)

**First time only — stage the SP8192 dataset to the Modal volume**:
```bash
modal run axes/quantization/pr1493_repro/modal_launcher.py --mode prefetch
```

**Training run** (~10 min on 8×H100 to hit PR-1493's 4550-step trajectory, plus a few min of Hessian collection):
```bash
modal run axes/quantization/pr1493_repro/modal_launcher.py \
  --mode train \
  --seed 42 \
  --run-id pr1493_bundle_seed42 \
  --iterations 20000 \
  --max-wallclock-seconds 600
```

This runs the full 600s budget. Training stops early if it hits the wallclock (as PR-1493 does at ~4550 steps). Post-hoc Hessian collection + bundle save adds ~15-30s.

**After the run — pull the bundle down**:
```bash
modal volume get parameter-golf-fineweb-cache \
  runs/pr1493_bundle_seed42/bundle \
  ./local_bundle
```

**Upload to HF for shared iteration**:
```bash
hf upload nprime06/parameter-golf-artifacts \
  ./local_bundle/ pr1493_seed42/
```

## Quantize a saved bundle on Modal (1×H100)

Reference reproduction — should produce a quantized BPB close to PR-1493's reported numbers:
```bash
modal run axes/quantization/pr1493_repro/modal_launcher.py \
  --mode quantize \
  --bundle-dir runs/pr1493_bundle_seed42/bundle \
  --run-id pr1493_quantize_reference
```

Variants (for Q1-Q11 experiments) — override any of:
```
--matrix-bits <int>          default 6
--embed-bits <int>           default 8
--matrix-clip-sigmas <float> default 12.85
--embed-clip-sigmas <float>  default 20.0
```

Example: probe whether `tok_emb` tolerates int6 at higher k (Q2):
```bash
modal run axes/quantization/pr1493_repro/modal_launcher.py \
  --mode quantize \
  --bundle-dir runs/pr1493_bundle_seed42/bundle \
  --run-id pr1493_q2_embed_int6_k20 \
  --embed-bits 6 \
  --embed-clip-sigmas 20.0
```

## Quantize locally (after `modal volume get` + HF download)

```bash
cd axes/quantization/pr1493_repro
BUNDLE_DIR=../../../local_bundle_seed42 \
DATA_DIR=/path/to/parameter-golf/data \
torchrun --standalone --nproc_per_node=1 quantize_bundle.py
```

Requires a local CUDA machine with flash_attn_3 installed (Hopper).

## Verifying before a full run

The launcher's `--mode train` with short iterations is a smoke test:
```bash
modal run axes/quantization/pr1493_repro/modal_launcher.py \
  --mode train \
  --iterations 50 \
  --max-wallclock-seconds 120 \
  --run-id pr1493_bundle_smoke
```

After it completes, the returned JSON should have `bundle_files` with all three names `exists: true`.

## Notes / caveats

- The Modal launcher disables TTT and ETLB (we only care about the pre-quant model). PR-1493's headline 1.0810 number includes TTT; our un-quantized ceiling will be a slightly higher number. That's expected — we're measuring the model weights, not the eval-time adaptation.
- Bundle is ~500-700 MB for 33M params (fp32 weights + all Hessians). `.gitignore` blocks `.pt` files from the repo.
- The first run after a rebuild of the Modal image will take a few extra minutes to build (flash_attn_3 install).
- If you hit `ModalVolumeStale` or similar, call `volume.reload()` — the launcher already does this at the start of each function.

## Reproducibility

Using `seed=42` with PR-1493's default hyperparameters (VOCAB_SIZE=8192, all defaults in `Hyperparameters` class) should hit the PR-1493 reported bundle characteristics. Deviations indicate a divergence from the reference.
