# Non-Record Submission: Reproduction of SOTA #1 (SmearGate + BigramHash + Int6 + SWA)

Reproduction of the March 20 SOTA entry (`Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`) on RunPod 8xH100 SXM, confirming reproducibility.

**Post-quant val_bpb: 1.1455** (published: 1.1458)

## Configuration

Identical to the original SOTA #1 submission — no modifications. Default env vars:

- 9 layers, 512 model dim, 8 heads, 4 KV heads (GQA)
- 3x MLP expansion (1536 hidden)
- SmearGate (learned bigram blending at embedding layer)
- BigramHash (4096 buckets, dim 128, projected to 512)
- Int6 per-row quantization + zstd-22 compression
- Muon optimizer: LR 0.02, momentum 0.99 (warmup 0.92 over 1500 steps), WD 0.04
- SWA: 30 checkpoints averaged during warmdown
- Sliding window eval (stride=64)

## Results

| Metric | Value |
|--------|-------|
| val_loss | 1.9341 |
| val_bpb | 1.1455 |
| Steps | 7365 / 20000 (wallclock capped at 600s) |
| Step avg | 81.47 ms |
| Artifact size | 15.88 MB |
| GPU | 8xH100 SXM (RunPod, IN datacenter) |

## Training Curve

| Step | val_loss | val_bpb |
|------|----------|---------|
| 0 | 6.9341 | 4.1068 |
| 500 | 2.3785 | 1.4087 |
| 1000 | 2.2447 | 1.3294 |
| 2000 | 2.1470 | 1.2716 |
| 3000 | 2.1037 | 1.2459 |
| 4000 | 2.0884 | 1.2369 |
| 5000 | 2.0580 | 1.2189 |
| 6000 | 2.0217 | 1.1974 |
| 7000 | 1.9759 | 1.1702 |
| 7365 (final) | 1.9341 | 1.1455 |
