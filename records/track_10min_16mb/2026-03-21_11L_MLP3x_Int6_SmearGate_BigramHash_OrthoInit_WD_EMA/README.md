# 11L MLP3x Int6+Zstd + SmearGate + BigramHash + OrthoInit + Muon WD + EMA

## Approach

combine established best practices identified through extensive ablation testing:

- **11 transformer layers** (optimal depth for the 10-min step budget)
- **3x MLP expansion** (1536 hidden dim, enabled by int6 compression savings)
- **Int6 quantization + zstd-22 compression** (fits ~33% more params vs int8+zlib)
- **SmearGate** (learned per-dim gate blending each token with predecessor)
- **BigramHash** (4096-bucket hash embedding for bigram context)
- **OrthoInit** (orthogonal weight initialization with proj scaling)
- **Muon weight decay 0.02** (decoupled, improves compressibility)
- **EMA 0.997** (exponential moving average of weights)
- **FP16 tied embeddings** (avoids quantization degradation on shared input/output projection)
- **Sliding window evaluation** (stride=256, maximal left-context scoring)
- **Sequence length 2048** for training and evaluation

## Key Findings from Ablation

During development, we systematically tested several novel approaches:

1. **Attention Residuals (AttnRes)**: Learned softmax attention over depth to replace residual connections. Both Full and Block AttnRes variants were tested. Result: compute overhead outweighs quality gains at this model scale (~54% slower per step with Full AttnRes).

2. **Depth recurrence (looped transformers)**: Weight sharing across layers with 2-6 loop iterations. Result: looping multiplies compute per step without adding parameters, reducing total training steps in the wall-clock budget.

3. **Sequence length curriculum**: Starting training with shorter sequences (256-512) for more gradient updates, ramping to 2048. Result: torch.compile recompilation overhead (~15s per shape transition) negates the step-count benefit.

4. **Test-time training (TTT)**: Full-parameter SGD on validation data before scoring. Result: only helps on well-converged models (0.001 BPB gain with 20-min trained model), actively hurts under-trained models.

5. **Undertrained larger model + TTT**: Using 13L instead of 11L with TTT as extended training. Result: 13L outperforms 11L given sufficient training steps (1.2375 vs 1.2444 at 20 min), but compression degrades with more training steps, creating a fundamental tension.

## Configuration

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Environment variables (all have defaults in the script):
- `NUM_LAYERS=11 MLP_MULT=3 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`
- `TRAIN_SEQ_LEN=2048 EVAL_STRIDE=256`
- `MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035`
- `WARMDOWN_ITERS=1200 GRAD_CLIP_NORM=1.0`
- `WEIGHT_DECAY=0.02 EMA_ENABLED=1 EMA_DECAY=0.997`
- `USE_SMEARGATE=1 USE_BIGRAM_HASH=1 USE_ORTHO_INIT=1`
- `FP16_EMBED=1 QUANT_BITS=6 USE_ZSTD=1`

## Results

3-seed validation on 8xH100 SXM:

| Seed | Steps | val_bpb | Artifact |
|------|-------|---------|----------|
| 1337 | 8611 | 1.1501 | 14,823,663 |
| 42 | 8825 | 1.1495 | 14,862,217 |
| 7 | 8827 | 1.1495 | 14,857,489 |

- **Mean val_bpb: 1.14971 ± 0.00036**
- **95% CI: [1.14930, 1.15012]**
- Training time: 600s (wallclock cap)
- Step avg: ~68ms on 8xH100
