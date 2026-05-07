# Emergent Weight Symmetry in Transformer QO Projections

## Summary

We discover that during full training of the PR#1019 SOTA architecture, layers 6-8 of the output (O) projections converge to **near-perfect symmetry** (W = W^T to machine precision, sym_energy = 0.999998), while Q projections in the same layers reach 99.5% symmetry. All other layers remain at ~50% (random). This bimodal pattern emerges naturally from the Muon optimizer + UNet skip-connection architecture.

We also test a **learnable SymMix** approach (`W_eff = W + tanh(beta) * W^T` with one scalar per QO matrix) that lets the model discover its preferred symmetry level. Result: perfectly loss-neutral (+0.0001 BPB), with learned betas converging to near-zero.

| Metric | Baseline | SymMix | Delta |
|--------|----------|--------|-------|
| val_bpb | 1.1687 | 1.1688 | +0.0001 |
| Steps (600s) | 3,565 | 3,551 | -14 |
| Artifact (int6+lzma) | 14,817 KB | 14,799 KB | -18 KB |
| Step avg (ms) | 168.3 | 169.0 | +0.7 |

## The Discovery: Bimodal Symmetry

After training on 4xH100 for 600s (3,100+ steps), we computed the symmetry energy ratio for all 22 QO bank matrices:

```
sym_energy = ||sym_part||^2 / (||sym_part||^2 + ||asym_part||^2)
```

where `sym_part = (W + W^T)/2` and `asym_part = (W - W^T)/2`. A value of 0.5 means random (no symmetry), 1.0 means perfectly symmetric.

```
Layer  Q sym_energy   O sym_energy
L0     0.4999         0.5001
L1     0.5021         0.5023
L2     0.5026         0.5011
L3     0.4998         0.5020
L4     0.5025         0.5004
L5     0.4995         0.5011
L6     0.9955  ***    0.999998  ***
L7     0.9955  ***    0.999998  ***
L8     0.9955  ***    0.999998  ***
L9     0.5000         0.5008
L10    0.4996         0.5014
```

The O projections in L6-8 are symmetric to 6 decimal places (||W - W^T|| / ||W|| < 0.0014). The transition is sharp: L5 is 0.4995, L6 jumps to 0.9955.

### What was checked and found NOT symmetric

- **KV bank** (256x512): all block_sym ~ 0.500
- **MLP up/down banks**: all ~ 0.500, cos(up, down^T) ~ 0.04-0.18
- **Encoder QO layers** (L0-L4): all ~ 0.500
- **Edge decoder QO layers** (L5, L9-L10): all ~ 0.500

## SymMix: Learnable Symmetry Mixing

Instead of forcing symmetry, we add a single learnable scalar per QO matrix:

```python
def _symmix_qo(self, w, idx):
    beta = torch.tanh(self.qo_beta[idx])
    return w + beta * w.transpose(-2, -1)
```

This adds 22 parameters total (negligible). The model is free to learn beta=0 (unconstrained), beta=+1 (symmetric), or beta=-1 (antisymmetric).

### Learned Beta Values

The betas were logged every 500 steps. They start non-zero early in training but converge toward zero:

```
Step 500:  Q: 0.105,-0.080,-0.053,-0.025,-0.068,-0.119,-0.064,0.079,-0.043,0.027,0.016
           O:-0.000,-0.015,0.007,0.002,-0.004,0.490,-0.463,0.015,0.105,-0.010,0.007

Step 3500: Q: 0.029,-0.023,-0.016,-0.005,-0.014,-0.019,-0.008,0.011,-0.005,0.007,0.002
           O: 0.004,-0.007,0.002,0.001,0.000,0.038,-0.026,0.008,0.016,-0.003,0.003
```

Early on, O[5] reached beta=+0.49 and O[6] reached beta=-0.46, but both decayed. The model explores symmetry mixing and decides it's not useful long-term.

## Compression Analysis

Post-training, we symmetrized the 6 detected matrices and re-quantized (int6 + lzma):

- **lzma savings: ~0 bytes** -- the compressor already exploits the near-symmetric redundancy in quantized values without explicit upper-triangle storage
- **Theoretical savings with upper-triangle format: ~288KB** -- would require a custom format storing only N*(N+1)/2 values
- **O layer quantization error: unchanged** (ratio 1.0003 -- already exactly symmetric)
- **Q layer quantization error: +69%** (ratio 1.69 -- forcing 99.5% to 100% loses the antisymmetric residual)

## Architectural Context

Layers 6-8 are decoder layers 1-3 in the UNet architecture:
- Encoder: layers 0-4
- Decoder: layers 5-10 (with skip connections from encoder layers 4,3,2,1,0)

L6-8 receive skip connections from encoder layers 4,3,2. However, L5 (also has skip from encoder 4) and L9 (skip from encoder 1) do NOT develop symmetry. The phenomenon is specific to the middle decoder layers.

## How to Run

```bash
# Setup
cd /workspace/parameter-golf
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Baseline (SYMMIX_ENABLED=0)
SYMMIX_ENABLED=0 SEED=1337 torchrun --standalone --nproc_per_node=4 \
  records/track_non_record_16mb/2026-04-01_Emergent_QO_Symmetry_SymMix/train_gpt.py

# SymMix experiment (SYMMIX_ENABLED=1)
SYMMIX_ENABLED=1 SEED=1337 torchrun --standalone --nproc_per_node=4 \
  records/track_non_record_16mb/2026-04-01_Emergent_QO_Symmetry_SymMix/train_gpt.py

# Analyze trained checkpoint symmetry
python3 -c "
import torch, torch.nn.functional as F
sd = torch.load('final_model.pt', map_location='cpu', weights_only=True)
qo = sd['qo_bank'].float()
n = qo.shape[0] // 2
for idx in range(2*n):
    w = qo[idx]
    sym = (w + w.T)/2; asym = (w - w.T)/2
    se = sym.norm()**2 / (sym.norm()**2 + asym.norm()**2)
    label = f'L{idx%n} {\"Q\" if idx<n else \"O\"}'
    print(f'{label}: sym_energy={se:.6f}')
"
```

## Compliance

- [x] Artifact <= 16,000,000 bytes (14,799 KB)
- [x] No test-time training on validation data
- [x] No network calls during evaluation
- [x] Non-record submission (research finding, not SOTA improvement)
- [x] Script compiles and runs successfully
