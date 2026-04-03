# NFT Quantization Field — Design Document

## Summary

Replace naive independent-rounding quantization with a GPU-native feedback loop inspired by NFT's 5-step adaptive measurement basis selection. The quantization field is not a separate data structure — it IS the weights, viewed through the quantization lattice. The GPU's native parallelism provides the higher-dimensional structure where interference between rounding decisions happens.

## NFT Loop Mapping

| NFT Step | Quantizer Implementation |
|---|---|
| 1. Spin outcome (measurement) | Forward pass with soft-quantized weights. The GPU reads the entire matrix-as-lattice-position in one matmul. |
| 2. Back-action (substrate changes) | The soft-quantized weights change the activations flowing through the model. The substrate is physically different. |
| 3. New rules (physics changes) | Backprop computes gradients through the soft-quantized weights. The loss landscape for all weights shifts simultaneously. |
| 4. New basis (adaptive measurement) | Gradients update the blend parameters (sigmoids) — shifting which lattice neighbor each weight favors. The error accumulates across the full matrix dimensionality, not per-weight. |
| 5. New outcome (next measurement) | Optimizer updates weights. They move relative to the grid. Next forward pass sees new lattice positions. Loop restarts. |

## Core Mechanism: Soft Quantization via Temperature-Controlled Sigmoid

For each weight w with quantization scale s:

```
grid_down = floor(w / s) * s
grid_up   = grid_down + s
distance  = (w - grid_down) / s          # in [0, 1]
blend     = sigmoid((distance - 0.5) / T) # temperature-controlled
w_soft    = (1 - blend) * grid_down + blend * grid_up
```

- At T = infinity: blend ≈ 0.5 for all weights (full superposition, maximum exploration)
- At T → 0: blend → 0 or 1 (collapsed, committed to a grid point)
- Gradients flow through sigmoid, so backprop tells each weight which grid neighbor reduces loss

The entire operation is matrix-shaped — computed over the full weight tensor in one GPU operation.

## Key Insight: The GPU IS the Higher Dimension

The GPU processes a 512x1024 weight matrix as ONE object in 524,288-dimensional space. A matmul is a single movement through that space. We don't simulate quantum mechanics — we exploit the fact that the GPU already operates in the full-dimensional space where interference between rounding decisions is visible.

Naive quantization snaps each coordinate independently to the nearest lattice point. But the nearest lattice point coordinate-by-coordinate is NOT the nearest lattice point in the full space measured by val_bpb. The NFT loop finds the better lattice point by moving through the full space.

## Training Protocol

### Phase 1: Parent Training
- Standard baseline training (9 layers, 512 dim, int8 target)
- Save checkpoint
- This is the pretrained "parent" — gives the child a viable starting body

### Phase 2: NFT Loop Training (the child)
- Load parent checkpoint
- Activate quantization field (soft quantization on all weight matrices)
- Temperature schedule: T starts high (broad exploration), decays toward 0 (commitment)
- Temperature follows learning rate decay — as optimizer calms down, distributions commit
- Continue training with standard optimizer + soft-quantized forward pass
- At end: T ≈ 0, hard-quantize by reading the collapsed sigmoids
- Evaluate val_bpb on the hard-quantized model

### Temperature Schedule
```
T(step) = T_max * (1 - step / total_steps) ^ power
```
- T_max: tunable, start with 1.0
- power: controls sharpening curve, start with 2.0 (slow start, fast finish)
- Could also tie to learning rate multiplier directly

## What We Measure (Overnight Run)

1. **Parent val_bpb (pre-quant):** How good is the float model?
2. **Parent val_bpb (naive int8):** How much does naive rounding damage it?
3. **Child val_bpb (NFT int8):** How much does the NFT loop preserve?
4. **Quantization damage = (2) - (1)** for naive, **(3) - (1)** for NFT

If (3) < (2), the NFT loop produces less quantization damage than naive rounding. The idea works.

## Implementation Target

- MLX on Mac (train_gpt_mlx.py)
- Modify the forward pass to use soft-quantized weights
- Add temperature parameter and schedule
- Add checkpoint save/load for two-phase training
- Keep the same model architecture, optimizer, data pipeline

## Relation to Existing Approaches

- **QAT (Quantization-Aware Training):** Injects random noise matching the quantization grid. The NFT loop is different — it injects STRUCTURED information from the feedback state, not random noise. QAT is "train the plant to survive transplanting." NFT is "grow the plant where it will live."
- **STE (Straight-Through Estimator):** Hard-quantizes in forward pass, pretends it didn't in backward pass. NFT uses soft quantization — the sigmoid IS differentiable, no pretending needed.
- **GPTQ/AdaRound:** Post-training optimization of rounding decisions. NFT does this DURING training, so the model co-adapts.

## Open Questions

- Optimal T_max and power for the temperature schedule
- Whether to apply the field to all tensors or only the large weight matrices
- Per-row vs per-tensor scale computation during the soft-quantization phase
- How many Phase 2 steps are needed for the distributions to converge
