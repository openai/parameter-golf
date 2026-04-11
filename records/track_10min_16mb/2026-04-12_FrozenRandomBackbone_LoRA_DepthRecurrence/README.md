# Frozen Random Backbone + LoRA Adapters

## Idea

What if you never serialized the backbone at all?

The core idea: initialize a full transformer backbone from a fixed random seed, freeze it, and only train small LoRA adapter matrices on top. At serialization time, the frozen weights cost **0 bytes** in the artifact because they're reconstructed deterministically from the seed. Only the adapter weights (plus embeddings and control scalars) need to be saved.

This fundamentally changes the size/quality tradeoff. Instead of cramming a 36M-param model into 16MB with aggressive quantization, I'm storing ~30M adapter parameters in 13.5MB with 2.5MB of headroom to spare.

## Architecture

- **11 layers, 512d, 8 heads, 4 KV heads, MLP 3x** (LeakyReLU(0.5)^2)
- **Frozen random backbone**: Each linear layer (Q, K, V, proj, MLP fc, MLP proj) has a frozen random weight matrix reconstructed from `seed=42 + layer_id`. Gaussian init with Kaiming std.
- **LoRA adapters**: Rank-304 adapter on every linear layer. `y = frozen(x) + scale * B(A(x))`. adapter_A is Gaussian init, adapter_B is zero init, scale is learnable (init 1.0).
- **Depth recurrence**: Layers 3-5 looped 2 additional times, creating 17 virtual layers from 11 physical. Activated at 35% of training.
- **XSA** (cross-segment attention) on all layers
- **U-Net skip connections** with learned gates between encoder/decoder halves
- **Partial RoPE** (16 dims with NTK scaling)
- **Parallel residuals** from layer 7 onward
- **LN scale factor** (1/sqrt(layer_idx+1))

## Training

- **Optimizer**: Muon (matrix params) + AdamW (embeddings, scalars). Row normalization, momentum warmup, WD=0.095.
- **Batch**: 786K tokens/step, seq_len=2048, warmdown_frac=0.72
- **EMA disabled**: EMA with decay 0.9965 destroys adapter weights at low step counts (~400 steps on 1xH100). The averaging drags adapter_B back toward its zero initialization. Disabling EMA was worth ~0.47 BPB.
- **Warmup**: 20 steps standard + 20 steps with looping enabled (for torch.compile to see both code paths), then weights/optimizer reset.

## Quantization

- **GPTQ with Hessians**: Collect Fisher information from 64 calibration batches, use for Hessian-weighted quantization ordering.
- **Int6 SDClip** (k=12.85) for adapter weight matrices, **int8** for embeddings
- **Brotli** compression (quality=11) with byte-shuffle preprocessing
- Small tensors (adapter_scale, q_gain, attn_scale, etc.) stored as float16 passthrough

## Results

Validated on 1xH100 (RunPod, 10-min wallclock). This is a single-GPU run, not the official 8xH100 setup -- I ran out of RunPod credits before completing the 8xH100 validation (the distributed run was working fine, hit step 1000 at 5.6M tok/s before the pod was killed).

| Eval mode | val_loss | val_bpb |
|-----------|----------|---------|
| Pre-quant (no EMA) | 3.3923 | 1.3133 |
| Quantized (int6+brotli) | 3.4547 | 1.3374 |
| Sliding window (stride=256) | 3.4149 | **1.3220** |

- **Steps**: 439 (1352ms/step on 1xH100)
- **Artifact**: 13,512,579 bytes (13.5MB) -- well under 16MB
- **Peak memory**: 70.3 GB / 80 GB
- **Quant gap**: +0.024 BPB (negligible)

For comparison, a standard (non-adapter) run with the same architecture at MLP=3 got **1.3213 BPB** sliding window on 1xH100 but required EMA which adds 0.12 BPB post-quant overhead at this step count. The adapter approach avoids this entirely.

## What I tried that didn't work

- **EMA on adapters**: Massive regression (+0.47 BPB). adapter_B starts at zero, and EMA averages toward that zero init for hundreds of steps. Just disable it.
- **AdamW TTT on adapters**: Implemented and tested. Hurt BPB by +0.009 at lr=0.0005/3 epochs. The learning rate probably needs to be much lower. The pipeline works but needs more tuning than I had credits for.
- **Higher MLP (4x)**: Fewer steps on 1xH100 due to overhead (413 vs 439 steps). MLP=3 was a better tradeoff.
- **MODEL_DIM=480**: Broke flash_attn (head_dim=60 not multiple of 8).

## Interesting findings

1. **Random backbone is surprisingly good** -- the frozen random weights provide decent features out of the box. The adapters only need to learn corrections.
2. **Depth recurrence reuses adapters** -- layers 3-5 share the same adapter weights across 3 passes per forward. Each adapter parameter gets 3x the gradient signal on those layers.
3. **Artifact efficiency** -- 13.5MB for a model that performs within 0.001 BPB of a full-rank 13.6MB model. The random backbone is essentially free storage.
4. **8xH100 scaling works** -- distributed training with adapters ran at 5.6M tok/s on 8xH100. With ~4200 steps (10x more than 1xH100), the model would likely improve substantially.

## What I'd do with more compute

- Run proper 8xH100 validation (3-seed)
- Tune AdamW TTT (lower LR, maybe 1 epoch)
- Try EMA with very low decay (0.9 or 0.95) at higher step counts
- Increase adapter rank to fill the 16MB budget (currently 2.5MB headroom)
- Experiment with per-layer rank allocation (higher rank for attention, lower for MLP)

## Command

```bash
# 1xH100
USE_RANDOM_ADAPTERS=1 ADAPTER_RANK=304 MLP_MULT=3 python3 train_gpt.py

# 8xH100
USE_RANDOM_ADAPTERS=1 ADAPTER_RANK=304 MLP_MULT=3 torchrun --nproc_per_node=8 train_gpt.py
```

## Files

- `train_gpt.py` -- full training script (1470 lines, readable)
- `train_1xh100_r304.txt` -- training log from 1xH100 run
- `submission.json`
