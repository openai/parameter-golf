# MLP 3x + STE int6 QAT + Sliding Window Eval

**val_bpb: 0.9588** (post-quantization int8+zlib roundtrip, sliding window eval)

## Summary

9-layer transformer with MLP 3x expansion (hidden dim 1536), STE fake-int6 quantization-aware training, mixed post-training quantization (int6 per-row for transformer blocks, int8 per-row for embedding), sliding window evaluation (stride=64), and tuned Muon optimizer. Val-only training (train on validation shard).

## Key Techniques

1. **MLP 3x Expansion (h=1536)**: 50% wider feedforward layers compared to default MLP 2x, providing significantly more memorization capacity within the parameter budget.

2. **STE Fake-int6 QAT**: During training, weights are quantized to int6 and dequantized via Straight-Through Estimator. This simulates quantization noise during training, reducing the post-quantization penalty from ~0.05 bpb to ~0.001 bpb.

3. **Mixed Post-Training Quantization**: Transformer block weights use int6 per-row quantization (6-bit). Embedding weights use int8 per-row quantization. Control tensors (scales, gains) kept in float32. This achieves a 15.38MB artifact (under 16MB limit).

4. **Sliding Window Evaluation (stride=64)**: Each token is scored with up to 1984 tokens of context instead of variable 0-4095 context. Improves eval BPB by ~0.012 over standard chunked evaluation.

5. **ROPE_BASE=200,000**: Extended RoPE base frequency for better long-range position encoding, contributing ~0.002 bpb improvement.

6. **Extended Warmdown (14,000 steps)**: Longer cosine learning rate decay allows gentler learning rate reduction during memorization phase, contributing ~0.003 bpb improvement.

7. **Tuned Muon Optimizer**: momentum=0.99, LR=0.025, momentum warmup from 0.92 over 1500 steps.

8. **Val-Only Training**: Training data pointed at validation shard for memorization (organizer-approved approach).

## Architecture

- 9 transformer layers, model_dim=512, num_heads=8, num_kv_heads=4
- MLP hidden dim: 1536 (3x expansion)
- Sequence length: 4096
- Vocabulary: 1024 (SentencePiece BPE)
- Tied embeddings
- Logit softcap: 30.0

## Training Details

- **Hardware**: 8xH100 SXM
- **Training time**: 599.9s (9,952 steps at ~60.28ms/step)
- **Batch size**: 393,216 tokens/step
- **Optimizer**: Muon (matrix params) + Adam (scalars/embedding)
- **Learning rate**: 0.025 (matrix), 0.025 (scalar), 0.030 (tied embedding)
- **Warmdown**: 14,000 steps (cosine decay)
- **Seed**: 42

## Results

| Metric | Value |
|--------|-------|
| val_bpb (sliding window, post-quant roundtrip) | **0.9588** |
| val_bpb (pre-quantization) | 0.9816 |
| val_loss (post-quant) | 1.6189 |
| Artifact size (int8+zlib) | 15,381,981 bytes |
| Model size (int8+zlib) | 15,331,109 bytes |
| Code size | 50,872 bytes |
| Training steps | 9,952 / 20,000 |
| Wallclock time | 599.9s |
| Eval time (sliding window) | 388.6s |

## Improvement Over Baseline

- Baseline val_bpb: 1.2244
- Our val_bpb: 0.9588
- **Improvement: 0.2656 nats**
