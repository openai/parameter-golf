# HyperGPT — val_bpb=8.218640

**Track**: 10 min / 16 MB
**Run**: 20260403_063554 | 8×H100 SXM 80GB
**Artifact**: 10.47 MB

## Approach

Generative Weight Synthesis: each transformer layer synthesizes its weights
from a shared bank plus a per-layer low-rank adapter.

    W = W_shared + exp(scale) × (A @ B.T)

This allows 16 layers in ~11.5 MB vs a standard transformer needing ~15 MB
for the same depth.

### Architecture
- 16 layers, d=512, 8 heads / 4 KV heads (GQA), seq=1024
- SharedBank: 2 groups × 7 weight matrices (early vs late layers)
- Per-layer adapters: rank-48 low-rank A/B + learnable scale
- Activation: LeakyReLU² with trainable negative slope
- Vocab: sp1024 (1024 BPE tokens)

### Training
- Optimizer: Muon (adapters) + Adam (shared bank, embed)
- LR schedule: time-aware warmup/peak/decay over 580s
- EMA weights for final evaluation
- 10-min competition run, ~1600 steps

## Reproduction

```bash
git clone https://github.com/yassienshaalan/parameter-golf
cd parameter-golf
bash runpod_run.sh   # requires RunPod GPU pod
```
