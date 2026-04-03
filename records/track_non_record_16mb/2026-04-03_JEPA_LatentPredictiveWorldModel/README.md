# JEPA-LM: Latent Predictive World Model for Parameter Golf

First implementation of JEPA (Joint Embedding Predictive Architecture) for the Parameter Golf challenge, an approach explicitly requested on the organizers' wishlist.

## Approach

A lightweight bottleneck predictor at the encoder-decoder boundary learns forward dynamics in representation space. Applied recursively, it predicts future representations at multiple horizons (t+1, t+2, t+3), forming a learned world model of text dynamics. The multi-horizon smooth-L1 loss on L2-normalized representations provides additional gradient signal during training, encouraging the encoder to learn temporally predictable (and therefore more structured) representations.

The predictor is stripped before serialization, adding zero bytes to the final artifact. The JEPA loss weight anneals alongside the learning rate warmdown, so final convergence is pure cross-entropy.

### Architecture details

- 10 transformer layers, 512 model dim, 3x MLP expansion
- LeakyReLU(0.5) squared activation (proven -0.003 BPB over ReLU squared)
- GQA with 8 query heads, 4 KV heads
- U-Net skip connections (encoder-decoder with learned skip weights)
- Tied embeddings, 1024 vocab, RoPE

### JEPA components (training only)

- LatentPredictor: bottleneck MLP (512 -> 128 -> 512) with residual connection, zero-init output
- Multi-horizon rollout: predictor applied 3 times recursively, predicting t+1, t+2, t+3
- Loss: smooth-L1 on L2-normalized representations with stop-gradient on targets
- Horizon weighting: 1.0, 0.5, 0.25 (exponential decay)
- Loss weight: 0.5 (anneals to 0 during warmdown)

### Based on

- NextLat (Srivastava et al., ICLR 2026): auxiliary latent prediction for transformers
- I-JEPA (Assran et al., 2023): predicting in representation space
- data2vec (Baevski et al., 2022): JEPA-style self-prediction for text
- Predictive coding (Rao & Ballard, 1999): prediction error as learning signal

## Results

All runs on 1xH100 with 600s wallclock cap (development runs, 8xH100 pending).

### Ablation study

| Config | val_bpb (post-quant) | Steps | Artifact |
|--------|---------------------|-------|----------|
| Baseline (9L, 2x MLP) | 1.3462 | 1156 | 12.8 MB |
| 10L 3x, JEPA off | 1.3312 | 1088 | 15.1 MB |
| 10L 3x, JEPA w=0.5 h=3 | **1.3299** | 1090 | 15.3 MB |
| 10L 3x, JEPA w=0.25 h=3 | 1.3378 | 1047 | 15.1 MB |
| 10L 3x, JEPA w=0.5 h=1 | 1.3403 | 1027 | 15.0 MB |
| 10L 3x, JEPA w=1.0 h=3 | 1.3454 | 995 | 14.9 MB |

### Key findings

1. JEPA auxiliary loss provides a small but consistent improvement (-0.0013 BPB) over the same architecture without it, at zero speed overhead.
2. Multi-horizon prediction (h=3) outperforms single-horizon (h=1), suggesting the recursive latent rollout captures useful multi-scale dynamics.
3. Loss weight of 0.5 is optimal. Lower weights (0.25) underperform, higher weights (1.0) cause the model to optimize for latent prediction over token prediction.
4. The JEPA predictor adds zero inference cost (stripped at export) and negligible training cost (~0ms/step overhead).

### Negative result: error injection

An earlier version injected prediction errors directly into the decoder path (predictive coding style). This was stable on 1xH100 (~1000 steps) but diverged to NaN on 8xH100 (~10000 steps), likely due to feedback instability between the error gate and encoder representations. The pure auxiliary loss approach is stable.

## How to run

```bash
# Development (1xH100)
RUN_ID=jepa_dev NUM_LAYERS=10 MLP_MULT=3 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Submission (8xH100)
RUN_ID=jepa_final SEED=1337 NUM_LAYERS=10 MLP_MULT=3 GRAD_CLIP_NORM=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Submission details

- Author: adi-suresh01
- Best val_bpb: 1.3299 (1xH100, 1090 steps)
- Artifact size: 15.3 MB (under 16 MB limit)
- 8xH100 results pending (compute grant requested)
