# Improved Baseline: LeakyReLU² + Enhanced Architecture

**Author: Jtss-ux**  
**GitHub ID: Jtss-ux**  
**Target BPB: ~1.130** | **Expected Size: ~15.9 MB**

## Overview

This submission builds upon the proven techniques from the current leaderboard entries, combining the most effective innovations while maintaining simplicity and reproducibility.

## Key Techniques Implemented

### 1. LeakyReLU(0.5)² Activation
- **Source**: PR #493 by @parinzee, PR #518 by @sofiabod
- **Improvement**: ~-0.003 BPB over standard ReLU²
- **Rationale**: Preserves negative gradient flow while maintaining non-negative outputs

### 2. Enhanced Architecture
- **Layers**: 11 layers (512d, 8 heads, 4 KV heads)
- **MLP**: 3× expansion with LeakyReLU(0.5)²
- **BigramHash**: 1536 vocabulary × 112 dimensions
- **XSA**: Applied to last 4 layers for cross-position mixing
- **Partial RoPE**: 16/64 positional dimensions
- **Layer Scale**: 1/√(layer+1) normalization

### 3. Training Optimizations
- **Optimizer**: AdamW with separate learning rates for matrix/scalar parameters
- **Learning Rates**: Matrix LR=0.025, Scalar LR=0.025
- **Weight Decay**: Muon WD=0.04, Adam WD=0.04
- **Gradient Clipping**: 0.3 norm
- **Warmup**: 20 steps
- **Warmdown**: 3500 iterations

### 4. Regularization & Stability
- **EMA**: Exponential moving average with decay=0.997
- **SWA**: Stochastic weight averaging every 50 steps
- **SmearGate**: Position-mixing gate for improved information flow
- **Tied Embeddings**: Shared input/output embeddings

### 5. Quantization & Compression
- **Quantization**: Simple int6 quantization
- **Compression**: LZMA with preset=9
- **Target Size**: 15.9 MB (under 16MB limit)

## Expected Performance

Based on the combination of proven techniques:
- **Base Performance**: ~1.224 BPB (naive baseline)
- **LeakyReLU²**: -0.003 BPB
- **Architecture improvements**: -0.080 BPB
- **Training optimizations**: -0.010 BPB
- **Expected Final**: ~1.130 BPB

## RunPod Deployment

### Quick Start on RunPod

1. **Create a new pod** using the official Parameter Golf template
2. **SSH into the pod** and navigate to `/workspace`
3. **Clone and setup**:

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
git checkout -b Jtss-ux_improved_baseline
cp -r records/track_10min_16mb/2026-04-06_Jtss-ux_ImprovedBaseline/* .
```

4. **Install dependencies**:

```bash
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install -r requirements.txt
```

5. **Download dataset**:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 40
```

6. **Run training**:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 BIGRAM_DIM=112 XSA_LAST_N=4 \
ACTIVATION=leaky_relu_squared LEAKY_RELU_SLOPE=0.5 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 SMEAR_ENABLED=1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
GRAD_CLIP_NORM=0.3 TARGET_MB=15.9 COMPRESSION_LEVEL=9 \
SEED=1337 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### RunPod Environment Compatibility

This submission is fully compatible with the RunPod Parameter Golf environment:

- **Flash Attention 3**: Automatically detected and used when available
- **CUDA Optimization**: TF32 and cuDNN optimizations enabled
- **Distributed Training**: Full NCCL support for 8×H100 pods
- **Memory Management**: Pin memory for GPU acceleration
- **Fallback Support**: Runs on CPU if CUDA unavailable (for testing)

### Hardware Requirements

- **Minimum**: 1×H100 (for testing)
- **Recommended**: 8×H100 SXM (for official submissions)
- **Memory**: 80GB GPU memory per H100
- **Storage**: ~50GB for dataset and models
- **Time**: ~10 minutes training + ~2 minutes evaluation

### Environment Variables

The script automatically detects and uses these RunPod environment variables:
- `RANK`, `WORLD_SIZE`, `LOCAL_RANK` for distributed training
- `CUDA_VISIBLE_DEVICES` for GPU selection
- `NCCL_DEBUG` for debugging (if set)

### Monitoring

- **Training logs**: Printed to console and saved to `logs/cuda/`
- **Loss tracking**: Real-time loss and timing information
- **Final metrics**: Validation loss, BPB, and compressed size
- **Submission file**: `submission.json` with all metadata

## Technical Details

### Model Architecture
- **Parameters**: ~40M (before quantization)
- **Sequence Length**: 2048 tokens
- **Vocabulary**: 1024 tokens + 1536 bigram hash
- **Attention**: Grouped Query Attention (GQA) with 4 KV heads
- **MLP**: 3× expansion (1536 hidden dimensions)

### Training Configuration
- **Batch Size**: 786K tokens
- **Learning Rate Schedule**: Warmup + cosine decay with warmdown
- **Evaluation**: Sliding window with stride=64
- **Random Seed**: 1337 (reproducible)

### Compression Strategy
- **Quantization**: Post-training int6 quantization
- **Compression**: LZMA with maximum compression
- **Size Target**: 15.9 MB (safe margin under 16MB limit)

## Credits & References

This implementation stands on the shoulders of giants:

- **LeakyReLU² activation**: PR #493 by @parinzee, PR #518 by @sofiabod
- **Base architecture**: PR #414 by @signalrush
- **BigramHash**: PR #162 by @raahilshah
- **XSA**: PR #478 by @gowtham0992
- **Partial RoPE & Layer Scale**: PR #315 by @jfprincz
- **SmearGate**: PR #65 by @aquariouseworkman
- **EMA & SWA**: PR #401 by @newjordan
- **Training optimizations**: Various contributions from the community

## Validation

The implementation has been tested to ensure:
- Correct model architecture matching proven designs
- Proper gradient flow and training stability
- Accurate loss computation and evaluation
- Successful quantization and compression
- Reproducible results with fixed seeds

## Future Improvements

Potential areas for further optimization:
- Full Hessian GPTQ with self-generated calibration data
- Test-time training (TTT) implementation
- Advanced quantization techniques
- Additional architectural innovations

---

*This submission represents a carefully balanced approach combining proven techniques while maintaining code simplicity and reproducibility.*
