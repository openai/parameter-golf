# 11L EMA + Int6 + XSA + LeakyReLU² + Partial RoPE

## Results (3-seed validation)

| Seed | val_bpb | artifact_bytes |
|------|---------|----------------|
| 42   | 1.13109 | 15,764,564     |
| 1337 | 1.13085 | 15,626,741     |
| 2024 | 1.13067 | 15,923,256     |
| **Mean** | **1.13087** | |
| Std  | 0.00017 | |

## Key Changes from Baseline

1. **11 layers** (up from 10), 512 dim, 8 heads / 4 KV heads (GQA)
2. **XSA (Exclusive Self Attention)** on last 4 layers for better representation
3. **LeakyReLU(0.5)²** activation — squared leaky ReLU with 0.5 negative slope
4. **Partial RoPE** — only 16/64 dims use rotary embeddings
5. **EMA weight averaging** (decay=0.997) for smoother final weights
6. **Int6 quantization** for all large weight matrices + zstd-22 compression
7. **Scale clamping fix** — clamp_min(1/clip_range) improves quantization quality
8. **Smaller batch size** (524288 tokens) to fit more training steps (~8200 steps in 600s)
9. **BigramHash(2048, dim=128)** token embeddings
10. **warmdown_iters=4500** for learning rate schedule
11. **Higher learning rates** (matrix_lr=0.025, scalar_lr=0.025)

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Built on SOTA baseline by @thwu1 (PR #180).
