# Experiment 002 — TTT param-filter sweep

**Date:** TBD (Day 2)
**Hypothesis:** Restricting TTT to the fp32 control surface (`scales`) beats `all` at fixed compute by adapting the post-GPTQ residual error without fighting the quantization grid.
**Baseline:** Experiment 001 reproduction (TTT on `all`, expected ≈ 1.0808)
**Cost:** 6 configs × ~2 min eval each on 2×H100 = ~12 min × $6/hr = ~$1.20 per single seed; budget ~$15 with rerun headroom

## Why this exists

The current TTT updates every parameter (~34M floats) on a 32K-token chunk × 3 epochs. The matrix params just came off a careful GPTQ rate-distortion fit — every TTT step pulls them off that grid for marginal data signal. Hypothesis: only the fp32 control surface (~38K floats) gives genuinely informative gradients on 32K tokens.

## Configs

All other hyperparameters held at PR #1493 SOTA values. Only `TTT_PARAM_FILTER` varies. We run on the **same checkpoint** (loaded from `final_model.int6.ptz`) so any difference is purely attributable to the TTT filter.

| Run ID | `TTT_PARAM_FILTER` | Floats updated | Expected sign |
|--------|---------------------|----------------|---------------|
| `f_all` | `all` | 34.5M | reference (reproduces 001) |
| `f_scales` | `scales` | ~38K | **primary hypothesis** — should ≈ tie or beat |
| `f_scales_embed` | `scales+embed` | ~4.2M | scales + tok_emb adaptation |
| `f_last3` | `last_n_layers:3` | ~7M | "fine-tune the head" approach |
| `f_attn` | `attn_only` | ~7M | attention-only adaptation |
| `f_mlp` | `mlp_only` | ~23M | MLP-only adaptation |

## Code

The patched script is `Opus/code/train_gpt_v1.py`. Same env-var surface as the SOTA, plus `TTT_PARAM_FILTER` (default `all`). To run only TTT on a saved checkpoint, point the script's `train_and_eval` at a pre-quantized artifact — see `Opus/scripts/run_ttt_only.sh`.

## Commands

```bash
# Day-1 checkpoint mounted at $CKPT
export CKPT=/workspace/artifacts/seed42.int6.ptz

for FILTER in all scales scales+embed last_n_layers:3 attn_only mlp_only; do
  TAG=${FILTER//:/_}; TAG=${TAG//+/_}
  TTT_ENABLED=1 TTT_PARAM_FILTER=$FILTER \
    SEED=42 \
    LOAD_CHECKPOINT=$CKPT \
    RUN_ID=opus_e002_${TAG} \
    torchrun --standalone --nproc_per_node=2 \
      Opus/code/train_gpt_v1.py 2>&1 | tee Opus/experiments/logs/002_${TAG}.log
done
```

(Note: `LOAD_CHECKPOINT` env var doesn't exist yet in `train_gpt_v1.py` — needs adding to the script as part of Day 1 setup. See "Day 1 follow-up" below.)

## Day 1 follow-up — checkpoint loading

The current SOTA always trains from scratch. To run TTT-only experiments cheaply, we need to short-circuit `train_and_eval` to skip training when a checkpoint is provided:

```python
# In train_and_eval, near the top:
if os.environ.get('LOAD_CHECKPOINT'):
    h.quantized_model_path = os.environ['LOAD_CHECKPOINT']
    eval_model = deserialize(h, device)
    # ... run sliding + TTT eval directly, skip train_model entirely
else:
    base_model, compiled_model = train_model(h, device, val_data)
    serialize(h, base_model, Path(__file__).read_text())
    eval_model = deserialize(h, device)
```

Add this after experiment 001 confirms reproduction.

## Result

Fill in after running:

| Run ID | `val_bpb_ttt` | Δ vs `f_all` | Eval time | Notes |
|--------|---------------|--------------|-----------|-------|
| `f_all` | | 0 | | reference |
| `f_scales` | | | | |
| `f_scales_embed` | | | | |
| `f_last3` | | | | |
| `f_attn` | | | | |
| `f_mlp` | | | | |

## Decision criteria

- **`f_scales` beats `f_all` by ≥0.001 nats single-seed** → promote to Experiment 003 (LR sweep on `scales`)
- **`f_scales_embed` beats `f_all` by ≥0.001 nats** → keep as alt; sweep its LR separately
- **No filter beats `f_all`** → kill the selective-TTT angle, pivot to mixed-bit GPTQ (Experiment 010)
- **`f_scales` ties `f_all`** → still interesting (faster TTT, same quality); explore higher LR (since fewer params per gradient step, can take bigger steps)
