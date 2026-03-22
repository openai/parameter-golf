# Record: 11L Frontier Stack + Value Residual + Gated Attention

**val_bpb: TBD** (sliding window stride=64, 3-seed mean) | **~15.6 MB** | 8xH100 SXM, 600s

## Approach

This submission combines the proven community frontier stack with two novel additions:

### Base Stack (from PRs #315, #374, #414)
- 11 layers, 512d, 8H/4KV, MLP 3x (relu²), U-Net skips
- SmearGate + BigramHash(2048) + OrthoInit
- XSA on last 4 layers
- EMA (decay=0.997) + Tight SWA
- Partial RoPE (16/64 dims)
- LN Scale (1/sqrt(layer+1))
- Late QAT (STE in final portion)
- GPTQ-lite post-training quantization
- Int6 + zstd-22, FP16 tied embedding
- Muon WD=0.04, momentum=0.99

### Novel Additions

**Value Residual / ResFormer** (arXiv:2410.17897): Layer-0 value vectors are cached and mixed (alpha=0.5) into all subsequent layers' attention values. This provides a residual pathway through the value stream, allowing deeper layers to access raw input representations. 18 learnable parameters. Dev-scale ablation (#413) showed -0.015 BPB.

**Gated Attention** (arXiv:2505.06708): Per-head sigmoid gate applied to attention output. Eliminates attention sink artifacts. ~37K parameters. Dev-scale ablation (#413) showed -0.003 BPB, stacking additively with Value Residual.

## Results

| Seed | Steps | Sliding BPB (s64) | Artifact |
|------|-------|--------------------|----------|
| 1337 | TBD | TBD | TBD |
| 42 | TBD | TBD | TBD |
| 2025 | TBD | TBD | TBD |

**Mean: TBD | Std: TBD**

## Ablations

| Config | BPB | Delta |
|--------|-----|-------|
| Full stack (this) | TBD | baseline |
| - Value Residual | TBD | TBD |
| - Gated Attention | TBD | TBD |
| - Both novel | TBD | TBD |

## Run Command

```bash
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## What Failed

(Document negative results here after experimentation)

## What I Learned

(Document key insights here)
