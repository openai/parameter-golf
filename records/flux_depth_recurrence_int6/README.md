# Depth Recurrence + SwiGLU + Int6 Quantization

**val_bpb: 1.2269** | Artifact: 15.67MB | 4xH100 SXM, 2337 steps in 10 min

## Approach

This submission stacks three key ideas to maximize model quality under the 16MB cap:

### 1. Depth Recurrence (4 unique blocks x 3 passes)

Instead of 12 unique transformer layers, we use 4 unique blocks repeated 3 times each, giving an effective depth of 12 at ~1/3 the parameter cost. The model reuses the same weights across passes, which means the compressed artifact only stores 4 blocks worth of parameters while getting 12 layers of computation during forward pass.

This was the single biggest architectural win. In local experiments on M4 Pro (MLX, 200 training steps), 4x3 recurrence at dim=768 matched 11 unique layers at dim=512 in bpb while compressing to a smaller artifact (12.05MB vs 13.89MB). The recurrent model gets more compute per parameter.

### 2. SwiGLU Activation

Replaced relu^2 FFN with SwiGLU (silu(gate) * up projection), keeping parameter count iso by adjusting the inner dimension. SwiGLU gave a consistent -0.02 bpb improvement across all tested configurations in local experiments.

### 3. Int6 Middle Layer Quantization

The core challenge: depth recurrence compresses poorly under int8 + zlib because the same 4 blocks are used 3 times, so there's no redundancy for zlib to exploit (compression ratio ~0.91 bytes/param vs ~0.26 for non-recurrent models).

To fit under 16MB with dim=736 (20M params), middle transformer blocks (blocks 1 and 2 out of 0-3) are quantized to 6-bit range [-31, 31] stored as int8. This reduces entropy per byte, giving zlib more to work with. The first and last blocks stay at full int8 [-127, 127] to preserve quality at the boundaries. This saved ~5MB compared to pure int8, bringing the artifact from ~18.5MB to 15.67MB.

### 4. Sliding Window Evaluation

At eval time, instead of non-overlapping chunks, we slide the context window by 64 tokens and only score the last 64 tokens per window. Every token gets near-full context (2048 tokens) instead of variable context (0-2048). This is a free bpb improvement at eval time with no training cost. Improved raw 1.2444 to final 1.2269.

## Architecture Details

| Parameter | Value |
|-----------|-------|
| Model dim | 736 |
| Attention heads | 8 (head_dim=92) |
| KV heads | 4 (GQA) |
| Unique blocks | 4 |
| Recurrence passes | 3 |
| Effective depth | 12 |
| MLP multiplier | 3x |
| Activation | SwiGLU |
| Vocab size | 1024 (SP) |
| Train seq len | 2048 |
| Total params | ~20M |

## Training Configuration

- Optimizer: Muon + Adam (embeddings)
- Learning rate: 0.003 (Muon), 0.006 (Adam)
- Batch tokens: 524,288
- Warmup: 20 steps
- Gradient clipping: 1.0
- Hardware: 4xH100 SXM (RunPod)
- Training time: 600s (wallclock cap)
- Steps completed: 2,337

## Results

| Metric | Value |
|--------|-------|
| Raw val_bpb (step 2337) | 1.2444 |
| After int8+zlib roundtrip | 1.2269 (with sliding window) |
| Artifact size | 15,671,310 bytes |
| Under 16MB cap | Yes (328,690 bytes headroom) |

## Research Process

Architecture search was conducted locally on an M4 Pro MacBook using MLX, running 13 automated experiments via a modified autoresearch loop. Key findings from the local search:

1. Depth recurrence (4x3) outperforms wide-shallow architectures at equivalent compressed size
2. SwiGLU consistently beats relu^2 by ~0.02 bpb
3. 4 passes (4x4) is worse than 3 passes (4x3) at 200 training steps -- the longer gradient path hurts convergence in the short training regime
4. Width beats depth: dim=768 with recurrence outperforms dim=512 with more unique layers

## Limitations and Next Steps

This is a non-record submission trained on 4xH100 (2,337 steps). An 8xH100 run would get ~5,100 steps at dim=736, estimated to reach 1.18-1.20 bpb. Additional improvements being explored:

- Val-finetuning phase (already coded and tested locally, -0.16 bpb on MLX)
- SP-4096 tokenizer for better compression
- Larger dim with more aggressive quantization (mixed int4/int6/int8)

## Run Command

```bash
INT6_MIDDLE_LAYERS=1 NUM_UNIQUE_BLOCKS=4 NUM_PASSES=3 MODEL_DIM=736 MLP_MULT=3 \
USE_SWIGLU=1 TRAIN_SEQ_LEN=2048 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
