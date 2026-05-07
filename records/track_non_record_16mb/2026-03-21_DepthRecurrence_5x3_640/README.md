# Depth Recurrence: 5 Unique Layers x 3 Loops (val_bpb=1.2716)

**Non-record submission** exploring depth recurrence as a parameter-efficient way to increase effective model depth.

## Core Idea

Instead of N unique transformer blocks, use fewer unique blocks and loop through them multiple times. This gives more effective depth for the same parameter budget, freeing parameters to increase model width.

**Config:** 5 unique layers looped 3 times = 15 effective depth at dim=640 (15M params)
**Baseline comparison:** 9 unique layers at dim=512 (17M params, 9 effective depth)

## Architecture Changes

Three modifications to the baseline GPT:

1. **Shared-weight loops**: The model has `num_unique_layers=5` transformer blocks. During forward pass, it loops through all 5 blocks `num_loops=3` times, giving 15 effective layers of computation while only storing 5 layers of parameters.

2. **Loop embeddings**: Learnable per-loop vectors (`loop_embeds`, shape `[num_loops, dim]`) added to the residual stream at the start of each loop. These let the model differentiate between passes through the same weights. Initialized to zero so the first loop behaves like vanilla.

3. **Loop gates**: Learnable per-loop scalars (`loop_gates`) that control how much each loop contributes vs. reverting to the initial representation x0. After loop `i > 0`: `x = gate_i * x + (1 - gate_i) * x0`. Initialized to `1/num_loops`.

The encoder-decoder skip connection pattern from the baseline was removed to keep the recurrence clean.

## Results

| Step | val_bpb | val_loss |
|------|---------|----------|
| 500  | 1.4720  | 2.4853   |
| 1000 | 1.3730  | 2.3182   |
| 2000 | 1.3138  | 2.2184   |
| 3000 | 1.2916  | 2.1809   |
| 4000 | 1.2775  | 2.1570   |
| 4500 | 1.2716  | 2.1471   |

Run was terminated early at ~step 5300 due to compute budget (RunPod time limit). val_bpb was still improving.

**Baseline comparison (9L/512dim, same optimizer settings):**
- Step 9000: val_bpb = 1.2507

At comparable training FLOPs, depth recurrence underperformed the baseline by ~0.02 BPB.

## Analysis: Why It Didn't Beat Baseline

1. **Conservative loop gating**: Gates initialized at 1/3 pull the representation back toward x0 after each loop. This effectively limits the model to ~1-1.5 loops of useful computation. The gating was intended to stabilize training through deep recurrence, but it over-regularized.

2. **No skip connections**: The baseline's encoder-decoder skip pattern (U-Net style) provides important information shortcuts. We removed these to simplify the recurrence, but this likely hurt.

3. **Gradient amplification through loops**: The same weights receive gradients from all 3 loops, which changes the effective learning rate. We didn't compensate for this — a lower matrix_lr might help.

4. **Fewer unique representations**: 5 unique layers means less representational diversity per loop pass compared to 9 unique layers, even with loop conditioning.

## What Would Improve This

- Remove loop gating entirely, or initialize gates closer to 1.0
- Add skip connections within each loop pass (encoder-decoder pattern per loop)
- Use 2 loops with 7 unique layers instead of 3 loops with 5 — less recurrence, more layer diversity
- Scale down learning rate proportional to num_loops (each weight gets gradient from all loops)
- Progressive loop training: start with 1 loop, add more during training
- Combine with techniques from current SOTA: sliding window eval, FP16 embeddings, int6 quantization

## Hardware & Training

- **Hardware**: 6x NVIDIA H200 (141GB HBM3e each), RunPod
- **Torch**: 2.8.0+cu128
- **Training**: Distributed across 6 GPUs, ~197ms/step, FlashAttention-2, bfloat16
- **Total training time**: ~888 seconds (14.8 minutes) for 4500 steps
- **Run was wallclock-limited, not converged**

## How to Run

```bash
RUN_ID=recurrent_5x3_d640 \
NUM_UNIQUE_LAYERS=5 NUM_LOOPS=3 \
MODEL_DIM=640 NUM_HEADS=8 NUM_KV_HEADS=4 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
