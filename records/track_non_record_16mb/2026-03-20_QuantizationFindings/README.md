# Non-record: Quantization Findings — SWA Reversal + Int5 Failure

## Summary

Two quantization findings from extensive experimentation with int6/int5 quantization and SWA (Stochastic Weight Averaging) on the parameter-golf baseline.

**Finding 1: SWA reverses the quantization gap** — After averaging 84 checkpoints, the int6+zstd roundtrip BPB is *lower* than the pre-quantization BPB. This is counterintuitive: quantization should always degrade quality, but SWA's weight smoothing eliminates quantization-sensitive outliers, making the quantized model better than any single checkpoint.

**Finding 2: Int5 quantization is catastrophic for undertrained models** — Applying int5 to MLP layers (while keeping attention at int6) increases the quantization gap from 0.3 to 1.4 BPB (4.5× explosion). This directly contradicts the hypothesis that mixed-precision int5/int6 saves space while maintaining quality.

## Architecture

Same as the community consensus: 11L MLP3x, 512d, 8/4 GQA heads, SmearGate, BigramHash, 27.2M params.

## Finding 1: SWA Reverses Quantization Gap

### Setup
- SWA averaging 84 checkpoints (every 50 steps from step 6481)
- Quantization: int6 + zstd compression

### Results
| Metric | Value |
|---|---|
| Pre-quant val_bpb (last checkpoint) | 1.5536 |
| Post int6+zstd roundtrip val_bpb (SWA-84) | **1.5164** |
| **Quantization "gap"** | **-0.0372** (negative = improvement!) |

### Interpretation
SWA smooths the loss landscape by averaging model weights, eliminating sharp minima and outlier parameters. These outliers are precisely what quantization degrades most severely. By removing them, SWA makes the model more "quantization-friendly" to the point where the quantized SWA model outperforms the unquantized single checkpoint.

This suggests **SWA and quantization are synergistic**, not antagonistic. More SWA checkpoints → smoother weights → smaller quantization sensitivity → possible negative quant gap.

## Finding 2: Int5 Quantization Catastrophe

### Setup
- Attempt to use int5 for MLP weights (larger tensors, more space savings)
- Keep attention weights at int6 (more sensitive to quantization)
- Model undertrained: 10,670 steps on 1×H100 (vs typical 13K+ on 8×H100)

### Results
| Precision | Quant Gap (BPB) | Model Size |
|---|---|---|
| int6 (all layers) | +0.30 (before STE-QAT) | 12.6 MB |
| int6 (with STE-QAT) | +0.04 | 12.6 MB |
| **int5 MLP / int6 attn** | **+1.40** | ~10.5 MB |

### Interpretation
The 4.5× gap explosion with int5 occurs because:
1. **Undertrained models have sharper weight distributions** — fewer training steps mean less smoothing, more outliers that int5 can't represent
2. **MLP layers are large but not redundant** — at 3× expansion (d=512 → 1536), MLP weights carry critical information that 5-bit precision destroys
3. **Space savings are illusory** — saving 2 MB in model size is worthless if the quality gap makes the model uncompetitive

This opposes the int5 approach discussed in PR #180 for models trained under compute constraints.

## Reproducibility

```bash
# Training with SWA (produces both findings)
SWA_STEP_TIME_MS=810 QUANT_BITS=6 MAX_WALLCLOCK_SECONDS=7500 \
MLP_MULT=3 NUM_UNIQUE_BLOCKS=11 NUM_LOOPS=1 \
MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MUON_WD=0.04 SWA_EVERY=50 \
USE_SMEAR_GATE=1 USE_BIGRAM_HASH=1 \
USE_STE_QAT=1 STE_QAT_START_FRAC=0.75 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3 \
python train_gpt.py
```

## Files
- `train_gpt.py` — training script (SWA, int6/int5 quantization)
