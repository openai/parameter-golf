# Nemotron-H Inspired Mamba-3 Hybrid + Hinge Point Depth Recurrence

**Non-record submission. First Mamba depth recurrence and first hinge-point multi-recurrence in the competition.**

## Summary

This submission explores a hybrid Mamba-3 / Transformer architecture inspired by NVIDIA's Nemotron-H, with a novel depth recurrence strategy focused on the U-Net hinge point. While the absolute bpb does not beat SOTA, the architectural insights and systematic ablation study provide new findings for the SSM track.

**Key result:** post-quant val_bpb = **1.4765** (1000 steps, 1xH100, SP1024, GPTQ int6+LZMA, 8.2MB artifact)

## Architecture

- **7 Mamba-3 SISO layers + 1 Attention layer** (8 physical layers)
- Mamba-3 config: d_state=64, expand=2, headdim=64, chunk_size=64, ngroups=1
- Attention: GQA with 8 heads, 4 KV heads, RoPE base=10000
- Attention placed at layer 4 (evenly spaced, Nemotron-H style)
- U-Net encoder-decoder with skip connections
- `torch.compile(dynamic=False, fullgraph=False)`

### Depth Recurrence (Novel)

**Hinge point multi-recurrence:** Layers 3 and 4 (the U-Net hinge) are repeated twice, creating 12 virtual layers from 8 physical layers with zero extra parameters.

```
Physical: [M0, M1, M2, M3, A4, M5, M6, M7]
Virtual:  [M0, M1, M2, M3, A4, M3, A4, M3, A4, M5, M6, M7]
                              ↑ hinge layers 3,4 repeated 2x
```

Recurrence is enabled at 35% of training (step 350/1000) to allow initial convergence without the overhead.

## Ablation Results

### Depth Recurrence (first-ever on Mamba layers)

| Config | val_bpb (2000 steps) | Virtual layers | vs no-recur |
|--------|---------------------|----------------|-------------|
| No recurrence | 1.2916 | 8 | — |
| Block recur 2,3 | 1.2851 | 10 | -0.0065 |
| Block recur 2,3,4 | 1.2830 | 11 | -0.0086 |
| **Hinge recur 3,4 x2** | **1.2824** | **12** | **-0.0092** |
| 4-layer recur 2,3,4,5 | 1.2864 | 12 | -0.0052 |
| Dual Attn@hinge | 1.2899 | 11 | -0.0017 |

**Finding:** Focused recurrence at the hinge point outperforms spread recurrence. Repeating hinge layers 2x (12 virtual) beats 4-layer 1x (also 12 virtual) by 0.004 bpb.

### Approaches Tested and Ruled Out

| Approach | Result | Finding |
|----------|--------|---------|
| Remove RoPE (ROPE_FRACTION=0) | +0.072 worse | Small models (26M) need explicit position encoding, unlike Jamba (1.3B) |
| Ternary Mamba (BitLinear 1.58-bit) | +0.397 worse | 26M params insufficient for ternary (literature confirms min ~1.3B) |
| Q-Mamba DSQ (A=FP16 + mixed precision) | +0.066 worse than standard GPTQ | Full Hessian GPTQ already handles SSM outliers well |

### Quantization

Standard Full Hessian GPTQ int6 with AR self-generated calibration data (from PR #1355 pipeline). LZMA-9 compression.

- Pre-quant val_bpb: 1.3948
- Post-quant val_bpb: 1.4765
- Quantization gap: 0.082
- Artifact size: 8.2MB (well under 16MB cap)

## Reproduction

### Setup (RunPod or Modal with H100)

```bash
# Install dependencies
pip install -r requirements.txt

# Additionally, Mamba-3 modules need to be copied from mamba3-release branch:
git clone --depth 1 --branch mamba3-release https://github.com/state-spaces/mamba.git /tmp/mamba3src
PKG=$(python -c 'import mamba_ssm,os; print(os.path.dirname(mamba_ssm.__file__))')
cp /tmp/mamba3src/mamba_ssm/modules/mamba3.py $PKG/modules/
cp -r /tmp/mamba3src/mamba_ssm/ops/triton/mamba3 $PKG/ops/triton/
cp /tmp/mamba3src/mamba_ssm/ops/triton/angle_cumsum.py $PKG/ops/triton/
rm -rf /tmp/mamba3src

# Download dataset
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

### Training (1xH100, ~17 min for 1000 steps + GPTQ)

```bash
RUN_ID=nemotron_hinge \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=8 \
NUM_ATTN_LAYERS=1 \
ATTN_PLACEMENT=even \
MAMBA3_D_STATE=64 \
RECUR_LAYERS=3,4 \
RECUR_MODE=block \
RECUR_REPEATS=2 \
RECUR_START_FRAC=0.35 \
ITERATIONS=1000 \
torchrun --standalone --nproc_per_node=1 train_nemotron_hybrid.py
```

### Training (8xH100, 10 min — pending compute grant)

```bash
# Same config but with:
# torchrun --standalone --nproc_per_node=8
# MAX_WALLCLOCK_SECONDS=600
# Expected: val_bpb ~1.25-1.30 post-quant
```

## Credits / Built On

- **PR #1355** (@mamba3-hybrid author): Mamba-3 Hybrid base, GPTQ pipeline, MuonEq-R optimizer
- **NVIDIA Nemotron-H** (arXiv 2504.03624): Hybrid architecture inspiration (92% SSM + 8% attention)
- **Mamba-3** (ICLR 2026, Gu et al.): SISO SSM with complex-valued states
- **PR #1204** (@sisovic): Depth recurrence concept (adapted from Transformer to SSM)
- **Q-Mamba, Mamba-PTQ, Quamba2**: Mamba quantization research informing our ablations

## Compute

All experiments run on Modal.com 1xH100 instances. Pending OpenAI compute grant for 8xH100 runs.
Total compute used: ~$30 Modal credits across 20+ experiments.

## What's Next

1. Full 8xH100 10-min run with best config (pending compute)
2. SP8192 tokenizer (expected ~0.05 bpb improvement)
3. Long-context evaluation (Mamba's O(n) advantage for 8K-32K eval)
4. Enable TTT and EMA for additional gains
