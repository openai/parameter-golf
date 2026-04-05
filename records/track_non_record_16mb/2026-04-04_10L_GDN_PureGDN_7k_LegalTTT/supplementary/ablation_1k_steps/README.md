# Ablation: PureGDN vs Softmax Attention vs Hybrid (1000 steps, 3 shards)

Short ablation experiments to isolate the contribution of the Gated DeltaNet
token-mixing mechanism vs standard softmax sliding-window attention.

## Experiments

| Experiment | Config | Arch | Token Mixing | Layers |
|------------|--------|------|-------------|--------|
| exp00_baseline | Model A | `A_PureGDN` | All GDN (10 layers) | 10× GatedDeltaNet |
| exp02_hybrid_1swa | Model D | `D_GDN_1SWA` | 10 GDN + 1 shared SWA | [GDN×5, SWA, GDN×5, SWA_shared] |
| exp08_pure_softmax | Model H | `H_PureSWA` | All SWA (10 layers) | 10× SlidingWindowAttention |

All experiments share: dim=512, 8 heads, SwiGLU 3× MLP, BigramHash(3072, 112)
with trigram, RMSNorm, tied embeddings, 1024 vocab, 1024 seq len.

## Training Configuration (from common_env.sh)

- **Steps:** 1,000
- **Warmdown:** 500 steps (50%)
- **Training shards:** 3 (subset of fineweb10B)
- **Batch tokens:** 131,072
- **GPU:** 1× A100 (gpuA100x4 partition)
- **EMA:** 0.997
- **SWA:** every 50 steps
- **Late QAT:** threshold 0.15

## How to Reproduce

```bash
# 1. Set up experiment directory
cd experiments/
./setup_experiment.sh exp08_pure_softmax

# 2. Submit SLURM job
sbatch exp08_pure_softmax/run.sbatch
```

## SLURM Jobs

| Experiment | SLURM Job ID | Status |
|------------|-------------|--------|
| exp00_baseline (PureGDN) | 55554336 | PENDING |
| exp02_hybrid_1swa (Hybrid) | 55554337 | PENDING |
| exp08_pure_softmax (Pure SWA) | 55554338 | PENDING |

## Notes

- Model H (Pure SWA) is the control experiment — same architecture but with
  softmax sliding-window attention replacing all GDN layers
- Model D (Hybrid) uses a shared SWA layer at positions 6 and 12 in a
  12-layer stack (Zamba-style weight sharing)
- exp02 enables XSA_EVAL=1 for cross-segment attention during evaluation
- Results will be added once jobs complete
