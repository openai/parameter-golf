# Ablation: Architecture & KV Sharing Experiments (1000 steps, 3 shards)

Short ablation experiments to isolate the contribution of:
1. The Gated DeltaNet token-mixing mechanism vs standard softmax attention
2. Adjacent-layer KV sharing with parameter redeployment into depth/width

## Experiments

### Set 1: Token Mixing Ablation

| Experiment | Config | Arch | Token Mixing | Layers |
|------------|--------|------|-------------|--------|
| exp00_baseline | Model A | `A_PureGDN` | All GDN (10 layers) | 10× GatedDeltaNet |
| exp02_hybrid_1swa | Model D | `D_GDN_1SWA` | 10 GDN + 1 shared SWA | [GDN×5, SWA, GDN×5, SWA_shared] |
| exp08_pure_softmax | Model H | `H_PureSWA` | All SWA (10 layers) | 10× SlidingWindowAttention |

### Set 2: Adjacent-Layer KV Sharing

| Experiment | Config | Arch | KV Sharing | Layers | Dim | Unique Params |
|------------|--------|------|-----------|--------|-----|--------------|
| exp09_kv_share | Model I | `I_KVShare` | stride 2 | 10 | 512 | ~27.3M |
| exp10_kv_share_deeper | Model J | `J_KVShare_Deeper` | stride 2 | 12 | 480 | ~29.4M |
| exp11_kv_share_wider | Model K | `K_KVShare_Wider` | stride 2 | 10 | 544 | ~30.0M |

KV sharing shares `k_proj`, `v_proj`, `k_conv1d`, `v_conv1d` between adjacent
GDN layer pairs (layers 0-1, 2-3, 4-5, etc.). This saves ~528K params per pair.

- **Model I:** Same as A but with KV sharing → saves 2.64M params (tests sharing effect)
- **Model J:** 12 layers, dim=480, KV share → near iso-param with A (tests depth)
- **Model K:** 10 layers, dim=544, KV share → near iso-param with A (tests width)

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
| exp09_kv_share (KV Share) | 55554658 | PENDING |
| exp10_kv_share_deeper (KV+Deeper) | 55554659 | PENDING |
| exp11_kv_share_wider (KV+Wider) | 55554660 | PENDING |

## Notes

- Model H (Pure SWA) is the control experiment — same architecture but with
  softmax sliding-window attention replacing all GDN layers
- Model D (Hybrid) uses a shared SWA layer at positions 6 and 12 in a
  12-layer stack (Zamba-style weight sharing)
- exp02 enables XSA_EVAL=1 for cross-segment attention during evaluation
- Results will be added once jobs complete
