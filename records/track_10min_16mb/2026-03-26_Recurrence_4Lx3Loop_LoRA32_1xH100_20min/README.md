# Depth Recurrence 4Lx3Loop + LoRA32 on 1xH100

## Results

| Seed | val_bpb | val_loss | bytes_total |
|------|---------|----------|-------------|
| 42   | 1.3752  | 2.3220   | 8,330,903   |

## Setup

Experimental depth recurrence architecture on a single H100 GPU for 20 minutes of training.

## Architecture

- 4 unique transformer layers shared across 3 loops (12 effective depth)
- Per-loop LoRA adaptation modules (rank 32) for loops 1-2
- Learned level signals per loop iteration
- dim=512, 8 heads (4 KV heads, GQA), MLP mult=3
- Mixed int5/int6 quantization-aware training
- BigramHash embeddings: 10240 buckets, dim=128
- SmearGate token blending
- Tied input/output embeddings
- SWA (start_frac=0.4, every 50 steps)

## Training

- Muon optimizer with Newton-Schulz orthogonalization
- LR: embed=0.03, matrix=0.02, scalar=0.02
- Weight decay: 0.04, warmdown: 3000 iters
- Gradient clipping: 0.3
- 1417 steps in 20 minutes on 1xH100 SXM
- Val progression: step500=1.4551, step1000=1.3643, step1417=1.3283 (pre-quant)

## Observations

- 11.6M params, artifact only 8.3MB (7.7MB headroom under 16MB cap)
- Quantization roundtrip causes 0.047 bpb degradation (1.3283 -> 1.3752)
- Shared weights under multiple loop contexts are harder to quantize
- Loss was still dropping at final step, suggesting more training time would help
- Higher LoRA rank and longer training are promising improvement directions
