# Hymba: Hybrid Attention + Mamba SSM for Parameter Golf

**First competitive non-transformer architecture in the competition.**

A 7-layer hybrid model that runs attention and Mamba SSM in parallel within each block, achieving **val_bpb ~1.18** on 8×H100 in 10 minutes — beating the naive transformer baseline (1.2244).

## Architecture

Each block runs two branches in parallel on the same input:

1. **Attention branch**: Standard GQA (8 heads, 4 KV heads) with RoPE + SDPA
2. **Mamba branch**: Selective scan SSM (`mamba-ssm`) with causal conv1d, gated projection

Outputs are merged with a learned weighted average: `sigmoid(α) · attn + (1 - sigmoid(α)) · mamba`, then projected through a shared output layer.

Key design: the input projection for K, V, and Mamba (x, gate) is fused into a single matmul for GPU efficiency. Q is projected separately to allow for future factorization.

## Key Findings

**Shallow models win at this compute budget.** The SSM branch makes each layer more powerful (attention for local precision, Mamba for long-range state), reducing the need for depth. Fewer layers = faster steps = more training in 10 minutes. The optimal depth (7L) is significantly shallower than the transformer baseline (9-11L).

**Training stability determines quantization quality.** With standard LR (0.04), the int6 quantization gap was 0.02-0.06 BPB. Reducing to LR=0.02 with aggressive cosine warmdown (3000 steps) produced smoother, more quantization-friendly weights — shrinking the gap to 0.02 without QAT. The warmdown phase alone accounts for ~0.06 BPB improvement as EMA weights converge.

**The Mamba branch adds minimal overhead on multi-GPU.** At 7 layers with MLP 4×, Hymba runs at ~85ms/step on 8×H100 (~7,000 steps in 10 min). The selective scan is O(T) and the per-layer Mamba params are small, so gradient sync overhead is negligible.

## Configuration

| Parameter | Value |
|---|---|
| Layers | 7 |
| Model dim | 512 |
| MLP multiplier | 4× |
| Attention heads | 8 (4 KV) |
| Mamba expand | 1 |
| SSM state size | 8 |
| Sequence length | 2048 |
| Quantization | int6 + zstd-22 |
| GPTQ-lite | Per-tensor clip search |
| Optimizer | Muon (matrix), Adam (scalar/embed) |
| Matrix LR | 0.02 |
| Warmdown | 3000 steps, cosine |
| EMA decay | 0.997 |
| Eval | Sliding window, stride=64 |

## Dependencies

```
pip install mamba-ssm causal-conv1d
```

## Running

```bash
SEED=1337 USE_HYMBA=1 HYMBA_EXPAND=1 HYMBA_SSM_STATE=8 \
NUM_LAYERS=7 MLP_MULT=4 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TRAIN_SEQ_LEN=2048 \
WARMDOWN_ITERS=3000 WARMDOWN_SHAPE=cosine \
EVAL_STRIDE=64 QUANT_BITS=6 GPTQ_LITE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

| Metric | Value |
|---|---|
| val_bpb (mean, 3 seeds) | **1.1828 ± 0.0036** |
| Seed 1337 | 1.1868 |
| Seed 42 | 1.1818 |
| Seed 7 | 1.1797 |
| Naive transformer baseline | 1.2244 |
| Artifact size | ~15.1 MB |
| Training time | 600s (8×H100) |
| Eval time | ~172s |
| Parameters | 27.2M |
| Training steps | ~7,050 |
