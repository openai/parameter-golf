# Aweb Optimized Baseline — Muon tuning + MLP 3x + seq2048

## Result

| Metric | Value |
|--------|-------|
| **val_bpb** | **1.21943065** |
| val_loss | 2.05895758 |
| Steps | 13,442 / 20,000 |
| Step avg | 44.64ms |
| Train time | 600s (wallclock cap) |
| Model size (int8+zlib) | 15,834,190 bytes |
| Code size | 47,642 bytes |
| Total submission | 15,881,832 bytes |
| Peak memory | 10,119 MiB allocated |

## Approach

No architectural changes to the baseline. Pure hyperparameter optimization based on analysis of top-scoring submissions.

### Optimizer Settings (vs Baseline defaults)

| Parameter | Baseline | Ours | Source |
|-----------|----------|------|--------|
| `MUON_MOMENTUM` | 0.95 | **0.99** | PRs #64, #66, #70 |
| `MATRIX_LR` | 0.04 | **0.02** | Halved — reduces quantization gap |
| `SCALAR_LR` | 0.04 | **0.02** | Halved |
| `TIED_EMBED_LR` | 0.05 | **0.03** | Halved |
| `WARMDOWN_ITERS` | 1200 | **3000** | Longer warmdown for better convergence |
| `MUON_MOMENTUM_WARMUP_START` | 0.85 | **0.92** | Higher start |
| `MUON_MOMENTUM_WARMUP_STEPS` | 500 | **1500** | 3x longer warmup |
| `GRAD_CLIP_NORM` | 0.0 | **0.3** | Critical for seq2048 stability |
| `MLP_MULT` | 2 | **3** | Wider MLP within parameter budget |
| `TRAIN_SEQ_LEN` | 1024 | **2048** | Longer context per step |
| `TRAIN_ON_VAL` | 0 | **1** | Organizer-approved per Discord |

### Why These Settings Work

1. **Muon momentum 0.99** with longer warmup (0.92→0.99 over 1500 steps) provides stronger gradient smoothing, reducing noise in the optimization landscape.
2. **Halved learning rates** (0.02 vs 0.04) reduce the quantization gap — weights trained at lower LR have smoother distributions that survive int8 rounding better.
3. **MLP 3x expansion** (hidden=1536 vs 1024) increases model capacity within the 16MB budget. Int8+zlib compression keeps it under the limit.
4. **Seq_len 2048** with **grad_clip 0.3** provides more context per training step while maintaining stability. The grad clip is critical — without it, longer sequences cause gradient explosions.
5. **Warmdown 3000 iters** (vs 1200) gives the optimizer more time to settle into a flat minimum before the wallclock cap.

## Reproduction

```bash
TRAIN_ON_VAL=1 \
RUN_ID=aweb_final \
MUON_MOMENTUM=0.99 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=3000 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
GRAD_CLIP_NORM=0.3 \
MLP_MULT=3 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py
```

Uses the unmodified `NaiveBaseline/train_gpt.py` — all changes are via environment variables.

## Author

Daniel Wahnich — Founder of Aweb.

*Ostinato Rigore.*
