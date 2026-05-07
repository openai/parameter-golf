# Int6 STE + NorMuon + SWA + Sliding Window Eval

**Mean val_bpb: 1.16019** (3-seed verified)

## Key Techniques

1. **Int6 STE (Straight-Through Estimator)**: Fake int6 per-row quantization (range [-31,31]) applied on every forward pass during training with straight-through gradient bypass. The model learns to cope with quantization noise throughout training, yielding only +0.002 bpb gap from quantization.

2. **NorMuon optimizer**: Row-normalized Newton-Schulz updates on top of Muon. Maintains per-row second-moment buffer for adaptive step sizes. Replaces standard Muon for more stable convergence.

3. **3x MLP width (1536 hidden)**: Enabled by int6 compression savings. The wider MLP provides significantly more model capacity within the 16MB artifact budget.

4. **FP16 tied embedding passthrough**: The embedding tensor (most quantization-sensitive) is stored in fp16, never quantized to int6. Protects the critical input/output projection.

5. **Sliding window evaluation (stride=64)**: Every scored token gets 960 tokens of preceding context. ~0.033 bpb improvement over standard non-overlapping evaluation.

6. **SWA (Stochastic Weight Averaging)**: Averages 7 checkpoints collected every 200 steps during the warmdown phase. Smooths the loss landscape for better generalization.

7. **Zstd-22 compression**: Replaces zlib for quantized weight compression. Better compression ratio yields more artifact headroom.

8. **U-Net skip connections**: Encoder-decoder structure with learnable per-layer per-dim skip weights.

9. **Optimizer tuning**: NorMuon momentum 0.99, beta2 0.95, matrix_lr 0.020, warmdown 3000 iters, momentum warmup 1500 steps from 0.92.

## Results

| Seed | val_bpb | Steps | ms/step | Artifact |
|------|---------|-------|---------|----------|
| 1337 | 1.16146 | 12357 | 48.55 | 15,045,740 |
| 42 | 1.15935 | 12351 | 48.58 | 15,053,489 |
| 7 | 1.15976 | 12336 | 48.69 | 15,157,415 |
| **Mean** | **1.16019** | | | |

## Architecture

- 9 transformer layers, 512 model dim, 8 attention heads, 4 KV heads (GQA)
- Vocab 1024 (SentencePiece BPE), seq len 1024, tied embeddings
- relu² activation (relu then square)
- RoPE with learnable Q gain (init 1.5), logit softcapping (30.0)
- RMSNorm, no bias in linear layers

## Dependencies

Standard PyTorch + zstandard for zstd compression.

## Acknowledgments

Architecture and techniques adapted from community contributions to the parameter-golf challenge, particularly the int6 STE, NorMuon, and sliding window evaluation approaches.
