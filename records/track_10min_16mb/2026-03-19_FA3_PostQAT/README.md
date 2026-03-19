# Int6+zstd MLP1500 + FA3 + Sliding Window + QAT

## Summary

Five stacked improvements on the NaiveBaseline (1.2244 BPB):

1. **Int6 mixed quantization + zstd**: MLP, Q, V, and output projection weights stored as 6-bit with per-row fp16 scales, compressed with zstd level 22. Fits 21.4M params in 16MB.
2. **MLP hidden=1500 (2.93x expansion)**: Wider MLP enabled by int6 space savings. Baseline uses 1024 (2x).
3. **Sliding window evaluation (stride=64)**: Every scored token sees ~960 context tokens. Major BPB improvement over non-overlapping eval.
4. **Flash Attention 3 (Hopper kernels)**: ~8% faster step times via native BSHD tensor layout.
5. **Post-training QAT (30s)**: STE-based fake int8 quantize/dequantize reduces quantization penalty.

## Architecture

- 9 layers, d_model=512, 8 heads, 4 KV heads (GQA)
- relu^2 MLP with hidden_dim=1500
- Tied embeddings, vocab=1024, train_seq_len=2048
- 21,446,728 parameters

## Hyperparameters

| Parameter | Value |
|---|---|
| matrix_lr (Muon) | 0.02 |
| muon_momentum | 0.99 |
| muon_momentum_warmup | 0.92 -> 0.99 over 1500 steps |
| warmdown_iters | 3000 |
| qk_gain_init | 1.7 |
| grad_clip_norm | 0.3 |
| post_qat_seconds | 30 |
| train_seq_len | 2048 |
| eval_stride | 64 |

## Results (8xH100 SXM, RunPod)

- Training: 9,473 steps in 570s (60ms/step) + 30s QAT
- Pre-quant BPB: 1.1832
- **Post-quant sliding window BPB: 1.1747**
- Compressed artifact: 15.98MB / 16.00MB
- Improvement vs baseline: **-0.0497 BPB**

## Quantization Strategy

- `.mlp.`, `.attn.c_q.`, `.attn.c_v.`, `.attn.proj.`: int6 per-row
- `.attn.c_k.`: int8 per-row (default)
- `tok_emb.weight`: fp16 passthrough
- Control tensors (scales, gains, norms): fp32 passthrough
- Compressor: zstd level 22
