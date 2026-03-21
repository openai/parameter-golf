# JoeProAI: SwiGLU + BigramHash + SWA + Int6/ZSTD-22

## Result
- **val_bpb: 1.1634** (int6+zstd-22 roundtrip)
- Verified on 8xH100 80GB, 10-minute wall-clock budget
- 9,937 steps in 600s (~60ms/step)
- Model size: 14,772,576 bytes (14.1 MB)

## Key Techniques
- SwiGLU FFN activation (replacing relu(x).square())
- BigramHash embeddings (4096 buckets, 128 dim)
- Stochastic Weight Averaging (SWA) every 50 steps from 50%
- Int6 quantization + zstd-22 compression
- Muon optimizer with weight decay 0.02
- 10 transformer layers, dim=512, 8 heads / 4 KV heads

## Hardware
- 8x NVIDIA H100 80GB HBM3 (Modal)
- Peak memory: 12,817 MiB per GPU

## Discovery Method
SwiGLU activation was discovered via GEPA (Genetic Evolution for Parameter Architecture),
an automated research harness that proposes and validates training optimizations.
26 single-GPU experiments identified SwiGLU as the best improvement, then verified at 8xH100 scale.
