# Dev Notes (for resuming work)

## Current Best: 1.783 BPB (post-quant)
Config: `NUM_UNIQUE_LAYERS=2 NUM_LAYERS=6 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=3 PER_LAYER_SCALES=1 GRAD_CLIP_NORM=1.0`
1.45M params, 90ms/step on Apple Silicon, 100M tokens training data

## Files
- `train_gpt_mlx_exp.py` — All experimental features: layer sharing, per-layer scales, repeat embeddings, sliding window eval, DEQ eval, FTLE tracking, QAT, nuclear norm, SwiGLU, bounded recurrence, Kronecker
- `train_gpt_submission.py` — CUDA script for H100: layer sharing + per-layer scales + Muon WD + label smoothing + eval knobs
- `make_mini_shards.py` — `python3 make_mini_shards.py --train-tokens N --val-tokens M --dst PATH`
- `EXPERIMENTS.md` — Full strategy, competition analysis, all results

## Quick Start
```bash
source .venv/bin/activate
python3 make_mini_shards.py --train-tokens 100000000 --val-tokens 100000 --dst ./data/datasets/fineweb_fast

DATA_PATH=./data/datasets/fineweb_fast ITERATIONS=50000 VAL_LOSS_EVERY=50000 \
  TRAIN_LOG_EVERY=3000 MAX_WALLCLOCK_SECONDS=540 TRAIN_BATCH_TOKENS=4096 \
  GRAD_ACCUM_STEPS=1 MLX_MAX_MICROBATCH_TOKENS=4096 WARMUP_STEPS=10 \
  NUM_UNIQUE_LAYERS=2 NUM_LAYERS=6 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=2 \
  MLP_MULT=3 PER_LAYER_SCALES=1 GRAD_CLIP_NORM=1.0 WARMDOWN_ITERS=1000 \
  RUN_ID=test python3 train_gpt_mlx_exp.py
```

## What Worked
- Layer sharing (2 blocks, depth 6) — same quality, 1/3 params
- MLP 3x > MLP 2x at this scale
- Per-layer scales + repeat embeddings — -0.012 BPB
- Grad clipping 1.0 — small consistent gain
- relu^2 > SwiGLU at tiny scale (sparsity helps)
- 4 heads > 8 heads at 256d (head_dim=64 sweet spot)
- More data + steps is the primary lever
- 1.45M params trained long > 3M params trained short at equal wall time
- Blocks become contractive after training (validated DEQ theory)
- FTLE identifies 40-60% cold rows (less sensitive to quantization)

## What Didn't Work
- Width > 256d on Apple Silicon — too slow per step
- DEQ extra eval with 1-2 blocks — degenerate fixed point
- Mixed 4/8-bit quant — too aggressive
- Bounded recurrence — too constrained
- SwiGLU, 8 heads, MLP 4x, higher LR — all worse or crashed

## Competition Meta (as of March 20, 2026)
Best: 1.1483 BPB. Stack: int6 + MLP3x + SmearGate + BigramHash + sliding window + zstd-22 + SWA + Muon WD.
Nobody combines depth recurrence with full meta — that's our angle.
PR #167 is open with clean layer sharing submission. PRs to study: #162, #135, #148.

## Next Steps
1. Get H100 compute → test our sharing inside the winning meta stack
2. Port PR #162's int6/SmearGate/zstd code + our sharing
3. Per-iteration LayerNorm (RingFormer) — each cycle gets unique LN
4. On H100: 3 shared, 512d, MLP3x, depth 9, full meta
5. DEQ extra eval with 3+ blocks (needs diversity for meaningful fixed point)

## Constraints
- 18GB RAM Mac — models <3M params, batch ≤4096
- Bash timeout kills at ~10 min — use MAX_WALLCLOCK_SECONDS=540
- .venv required: `source .venv/bin/activate`
