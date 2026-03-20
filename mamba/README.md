# Mamba-2/SSD + Sparse Attention Hybrid LM

Hybrid autoregressive language model combining Mamba-2 (SSD) blocks with sparse attention layers for the OpenAI parameter-golf challenge.

## Architecture

- **10 total layers**: 8 Mamba-2 blocks + 2 attention blocks (at positions 7, 9)
- **Mamba-2 blocks**: expansion=2 (d_inner=1024), state_dim=16, 8 heads, conv_kernel=4
- **Attention blocks**: 8 heads, 4 KV heads (GQA), RoPE, relu² MLP (2x expansion)
- **Shared**: vocab_size=1024, d_model=512, seq_len=1024
- **Features**: Tied embeddings, logit softcap=30, RMSNorm, U-Net skip connections
- **~19M params**, estimated artifact ~15.5MB int8+zlib

## Simplifications vs Canonical Mamba-2

| Feature | Canonical Mamba-2 | This Implementation |
|---------|------------------|---------------------|
| A parameter | Diagonal matrix per head | Scalar per head |
| dt (timestep) | Input-dependent projection | Constant per head (learned bias + softplus) |
| Scan | Custom CUDA kernel with hardware-aware chunking | Sequential loop within chunks (chunk_size=64) |
| Conv1d | Fused with scan | Standard nn.Conv1d |

These simplifications trade some expressivity and speed for implementation simplicity. The mixer core is modular — swap in a more efficient scan implementation later.

## Usage

### Training (on GPU)
```bash
cd parameter-golf
python -m mamba.train
```

### Running tests (locally)
```bash
cd parameter-golf
python -m pytest mamba/tests/ -v
```

### Evaluation only
```bash
python -m mamba.eval  # (requires integration with saved model)
```

## Ablation Guide

### All-attention (baseline equivalent)
```bash
ATTN_LAYER_INDICES="0,1,2,3,4,5,6,7,8,9" python -m mamba.train
```

### All-Mamba (no attention)
```bash
ATTN_LAYER_INDICES="" NUM_LAYERS=10 python -m mamba.train
```
Note: set `ATTN_LAYER_INDICES` to empty string or a value outside layer range.

### Different attention placements
```bash
# Attention at layers 4 and 9 (middle + end)
ATTN_LAYER_INDICES="4,9" python -m mamba.train

# Attention every 3rd layer
ATTN_LAYER_INDICES="2,5,8" python -m mamba.train
```

### Adjust model depth/width
```bash
# Deeper but narrower
NUM_LAYERS=14 MODEL_DIM=384 python -m mamba.train

# Wider but shallower
NUM_LAYERS=8 MODEL_DIM=576 python -m mamba.train
```

### SSM parameters
```bash
# Larger state dimension
SSM_STATE_DIM=32 python -m mamba.train

# More SSM heads
SSM_NUM_HEADS=16 python -m mamba.train
```

## File Structure

```
mamba/
├── config.py      # Hyperparameters dataclass with env var overrides
├── model.py       # All model components (Mamba2Mixer, AttentionBlock, MambaHybrid)
├── data.py        # TokenStream, DistributedTokenLoader
├── eval.py        # val_bpb evaluation
├── train.py       # Main training script
├── README.md      # This file
└── tests/
    ├── test_scan.py       # Chunked scan vs naive sequential
    ├── test_model.py      # Shapes, gradient flow, param count
    └── test_causality.py  # No future-token leakage
```
