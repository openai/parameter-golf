# 10L Optimized

10-layer transformer with mixed quantization and training improvements.

## Results

| Seed | val_bpb |
|------|---------|
| 42   | 1.1477  |

Artifact size: 15.75 MB

## Architecture

- 10 layers, 512 dim, 8 heads / 4 KV heads (GQA)
- 3x MLP expansion with improved activations
- Hash-based token embeddings
- U-Net skip connections, tied embeddings
- Mixed int5/int6 quantization + zstd-22

## Training

- Muon optimizer + AdamW
- EMA weight averaging
- 10 minutes on 8xH100 SXM
