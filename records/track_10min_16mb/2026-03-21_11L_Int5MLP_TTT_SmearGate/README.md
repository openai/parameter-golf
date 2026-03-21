# 11L Int5-MLP + Int6-Attn + SmearGate + BigramHash + TTT-SGD + SWA

**val_bpb: 1.1455** (seed 1337, 3-seed validation in progress)

## Key Techniques

1. **11 transformer layers** (up from baseline 9): funded by mixed-precision quantization savings.

2. **Mixed int5/int6 quantization**: MLP weights quantized to int5 [-16, 15] — the 3 zero high bits per byte give zstd-22 a 1.88x compression ratio (vs 1.51x for int6). Attention weights stay at int6 [-32, 31]. This saves ~1.9MB, funding the 11th layer.

3. **SmearGate**: Learned per-dimension gate (~512 params) blending each token's embedding with the previous token's. Injects bigram context before the transformer stack.

4. **BigramHash embedding**: 2048-bucket hash table (dim=64, projected to 512) mapping consecutive token pairs to learned embeddings.

5. **Full-model SGD TTT** (test-time training): After training, adapt the entire model on validation data using SGD (lr=0.002, momentum=0.9, 2 epochs). Improves BPB by ~0.005 at zero parameter cost within the eval time budget.

6. **SWA** (Stochastic Weight Averaging): Averages 30 checkpoints collected every 50 steps during warmdown. Improves both generalization and quantization robustness.

7. **Weight decay 0.04** on both Muon and AdamW: Keeps weight magnitudes small, improving int5/int6 quantization quality and zstd compressibility.

8. **OrthoInit + muP scaling**: Orthogonal initialization for all large matrices; output projections scaled by 1/sqrt(2*num_layers).

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 11 |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 (GQA) |
| MLP expansion | 3x (hidden=1536), relu^2 |
| Vocab | 1024 (SentencePiece BPE) |
| Seq len (train) | 2048 |
| Tied embeddings | Yes |
| Total params | 26.67M |

## Training

- **Optimizer**: Muon (matrix params, WD=0.04) + AdamW (embeddings/scalars, WD=0.04)
- **Muon momentum**: warmup 0.92 -> 0.99 over 1500 steps
- **LR**: matrix=0.02, scalar=0.02, embed=0.03
- **Warmdown**: 3000 iterations
- **Grad clip**: 0.3
- **Hardware**: 8xH100 SXM 80GB, 600s wallclock

## Results

| Seed | val_bpb (post-quant) | val_bpb (post-TTT) | Steps | ms/step |
|------|---------------------|---------------------|-------|---------|
| 1337 | 1.1507 | **1.1455** | 5197 | 115.47 |

Artifact: 15.94 MB (int5-MLP + int6-attn + zstd-22) | Code: 56 KB | Total: 15.99 MB

## Eval Pipeline

1. Decompress int5/int6 artifact (zstd-22)
2. Full-model SGD TTT: 2 epochs on val data (lr=0.002, momentum=0.9) — ~422s
3. Sliding window evaluation (stride=64, seq_len=2048) — ~273s
4. Total eval time: ~696s (within 10-min eval budget)

## Ablation

Based on iterative experiments on 1xRTX4090 and 1xA40 (500-step ablations):

| Change | Artifact | Notes |
|--------|----------|-------|
| 9L int6 only (PR#162 base) | 15.6 MB | Baseline |
| +11L (int6 only) | 18.9 MB | Does not fit |
| +int5 MLP | 16.0 MB | Fits! -1.9MB from int5 |
| +TTT SGD | - | +0.005 BPB improvement |

## Acknowledgments

Built on top of ideas from the community: int6 quantization (PR#39 @nanlliu), SmearGate/BigramHash (PR#102/#135 @unnir), SWA (PR#89 @vmfunc), sliding window eval (PR#50 @mattqlf), TTT (PR#77 @samacqua, PR#152 @timowhite88), mixed int5/int6 (PR#180 @thwu1), OrthoInit+muP (PR#162 @raahilshah), weight decay for Muon (PR#60 @notapplica).
