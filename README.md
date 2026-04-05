# Tiny GPT Submission (GLU + GQA)

This repository is prepared for a reproducible parameter-golf submission run using a compact Tiny GPT model with a GLU MLP and Grouped Query Attention (GQA).

## Final Model

- Model: Tiny GPT (character-level)
- Attention: GQA with `num_heads=2`, `num_kv_heads=1`
- MLP: GLU-style gated MLP
- Positional encoding: sinusoidal + RoPE in attention
- Parameter count: **1536**

## Why GQA Helped

Grouped Query Attention improved optimization by allowing two query heads to specialize while sharing a single K/V head. In this tiny-parameter regime, that improves representational efficiency without large parameter growth, leading to lower validation loss than the previous single-head baseline.

## Training Setup

- Script: `python train_gpt.py`
- Fixed mode: `char_small`
- Seed: `1337`
- Steps: `10000`
- Batch size: `16`
- Sequence length: `32`
- Optimizer: AdamW
- LR: 0.002 with cosine decay and warmup
- Warmup steps: 250
- Min LR: 1e-5
- EMA: enabled (`decay=0.999`)
- Label smoothing: removed from final submission path

## Reproducibility

The final defaults are pinned in `train_gpt.py` so that running the script directly reproduces the submission configuration:

```bash
python train_gpt.py
```

At startup, the script prints:

- `model_params:<count>`
- full `char_train:start ...` configuration line

At completion, the script prints and saves:

- `char_best best_val_loss=... best_step=...`
- `char_validation loss=... ppl=...`
- `logs/submission_metrics.json`

## Final Results

Best run (see `logs/gqa_2h1kv_10k_console.log`):

- Best validation loss: **2.7382**
- Final validation loss: **2.7570**
- Final perplexity: **15.75**
- Parameters: **1536**
- Training steps: **10000**

## Logs Kept for Submission

- `logs/gqa_2h1kv_10k_console.log` (best run log)
- `logs/submission_metrics.json` (structured summary from final code path)

## Notes

- Experimental branches (quick A/B and label smoothing switches) were removed from the active training path to keep the submission code concise and reproducible.
- The code still preserves core project functionality while defaulting to the final submission configuration.
