## Progressive Depth 4-Hour: Depth Recurrence Scaling Study

val_bpb = **1.0889** (Hedge Mixer) | 1.1271 (sliding) | 1.1613 (roundtrip)

4 hours on 8xH100 SXM. Non-record submission — unlimited compute track.

### Research Question

How does depth recurrence scale with compute? This is the first data point.

Shared-weight recurrence has a unique scaling property: the same 3 blocks receive every gradient update. At 132K steps with 5 repeats, each block saw **~660K effective gradient passes** — impossible with unique-layer architectures at the same parameter count.

### Scaling Curve

| Steps | Time | Phase | val_bpb | Delta from prev |
|-------|------|-------|---------|-----------------|
| 5K | 6 min | 2 rep | 1.3061 | — |
| 30K | 36 min | 2 rep | 1.2713 | -0.035 |
| 55K | 66 min | 2 rep | 1.2649 | -0.006 |
| 60K | 73 min | 3 rep | 1.2627 | -0.002 |
| 85K | 117 min | 3 rep | 1.2437 | -0.019 |
| 100K | 151 min | 4 rep | 1.2341 | -0.010 |
| 115K | 188 min | 5 rep | 1.2273 | -0.007 |
| 125K | 217 min | 5 rep | 1.2179 | -0.009 |
| 132K | 240 min | 5 rep + SWA | **1.1576** | -0.060 |

Key observations:
1. **Phase transitions matter**: each depth increase gives an immediate improvement, even late in training
2. **SWA is massive at scale**: 38 checkpoints gave -0.060 bpb — larger than any single phase transition
3. **Diminishing returns within phase**: Phase 1 (2 rep) shows clear flattening after ~40K steps
4. **Progressive Depth unlocks 5 repeats**: 15 effective layers from 3 physical blocks, only possible with gradual depth ramp

### Comparison

| Run | Compute | Steps | Sliding bpb | Hedge bpb |
|-----|---------|-------|-------------|-----------|
| Will DePue baseline (flat 9×512) | 4 hours | 329K | — | 1.2074 |
| Our 10-min Progressive Depth | 10 min | 5.7K | 1.1966 | 1.1454 |
| **Our 4-hour Progressive Depth** | **4 hours** | **132K** | **1.1271** | **1.0889** |

4-hour depth recurrence beats 4-hour flat baseline by **0.119 bpb** (1.2074 → 1.0889 with Hedge, comparison without Hedge: 1.1271 vs 1.2074 = **-0.080**).

### Configuration

```bash
MAX_WALLCLOCK_SECONDS=14400 ITERATIONS=200000 \
NUM_REPEATS=5 PROG_DEPTH="0.3:2,0.5:3,0.75:4,1.0:5" \
WARMDOWN_ITERS=15000 SWA_EVERY=100 \
MATRIX_LR=0.015 SCALAR_LR=0.015 TIED_EMBED_LR=0.018 \
VAL_LOSS_EVERY=5000 TRAIN_LOG_EVERY=1000 USE_HEDGE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Architecture

3 shared blocks, progressive depth (2→3→4→5 repeats), dim=832, 8 heads, 4 KV heads, GQA, MLP 2×, tied embeddings. Cross-Repeat Skip (#148), XSA, LeakyReLU², SWA, Hedge Mixer. 17.15M params, 15.82MB artifact.

### Credits

Hedge Mixer from PR #688 (@RoyiRa), PR #745 (@stukenov). Will DePue's 4-hour baseline for comparison.
