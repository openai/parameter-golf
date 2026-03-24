# Radial Mixed Bridge 16MB

This submission introduces the **Radial Mixed Bridge** architecture, a dual-branch design optimized for high-capacity representation under the Parameter Golf 16MB constraint.

## Architecture: Dual-Branch Bridge

Unlike standard monolithic transformers, the Mixed Bridge employs two parallel compute branches:
- **Branch A:** 8 layers, 384 hidden dimension. Optimized for deep feature extraction.
- **Branch B:** 5 layers, 320 hidden dimension. Optimized for lower-frequency structural patterns.

The results from both branches are fused into a 448-dimensional space before the final LM head.

### Key Innovations

1. **Mixed Precision Export & Pruning**
   - We employ a heterogeneous quantization strategy: **INT6** for bridge projections and **INT8** for core attention and MLP layers.
   - **Pruning**: Weights with absolute values below `0.0025` are Zero-Pruned before compression, significantly increasing the information density per byte.

2. **FROStable Optimizer + EMA**
   - **FROStable**: An evolution of the Fractal Resonant Optimization, specifically tuned for multi-branch stability under mixed quantization.
   - **EMA (Exponential Moving Average)**: A decay of `0.997` is applied during training. The final submission uses the EMA-averaged weights, which drastically improves the BPB stability post-quantization.

3. **Radial Encoding**
   - Both branches leverage pure geometric `RadialEncoding(8)`, eliminating the need for bulky positional embedding tables.

## Results

- **Val BPB:** 1.8720
- **Parameters:** 18,015,808
- **Artifact Size:** 15,129,854 bytes (Pass ✅)

## Configuration

- **Branches:** 2 (Dual)
- **Total Layers:** 13 (8+5)
- **Fused Dim:** 448
- **Target Wallclock:** 600s

## Reproducibility

The `train_gpt.py` script follows the rigorous record-track compliance established in our primary submission, including full artifact auditing and distributed NCCL support.
