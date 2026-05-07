# Phase 1.2: 12L Leaner Architecture (480 hidden)

## Overview
Tests whether adding depth (11L → 12L) with reduced width (512 → 480) can improve loss while staying within 16MB artifact limit.

## Hypothesis
- Deeper models can capture more complex patterns
- Width reduction (512→480) saves ~2-3% params, compensated by +1 layer
- Target: Loss ≤ 1.1200 with model size ≤ 15.0 MB

## Key Configuration Changes
```python
# From best record (1.1271):
# - num_layers = 9  → num_layers = 12
# - model_dim = 512 → model_dim = 480
# - mlp_mult = 3.0  → mlp_mult = 3.0 (unchanged, now 3×480=1440 hidden)
# - All other settings identical
```

## Difference from Baseline (1.1271)
- **Layers**: 11 → 12 (split as 6 encoder + 6 decoder + skip connections)
- **Hidden dim**: 512 → 480
- **MLP expansion**: 1536 (512×3) → 1440 (480×3)
- **Vocabulary embedding**: tied, 2×480×1024 (vs 512×1024)
- **Total params**: ~3-5% reduction overall due to width offset

## Quantization
- Same int6 STE QAT (31 levels: -32 to 31)
- Per-row quantization for attention/MLP layers
- zstd-22 compression
- Expected model size: **~15.0 MB** (1 MB buffer maintained)

## Training Configuration (Identical to Best Record)
- **Duration**: ~10 minutes on 8×H100
- **Batch**: 786,432 tokens/step, seq_len=2048
- **Optimizer**: Muon (matrix) + Adam (scalars/embeddings)
- **Learning rates**:
  - Tied embeddings: 0.05
  - Matrix (Muon): 0.04, WD=0.02
  - Scalars (Adam): 0.04, WD=0.01
- **EMA**: Yes, decay 0.997
- **XSA**: On last 4 layers
- **SmearGate + BigramHash**: Enabled

## Expected Outcome
- **Best case**: ≤1.1180 (good depth-width tradeoff)
- **Moderate**: 1.1200-1.1220 (marginal gain from depth)
- **Worst case**: >1.1270 (if depth doesn't help with tight params)

## Implementation Notes
1. Copy best record `train_gpt.py` as base
2. Change line: `NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 11))` → `12`
3. Change line: `MODEL_DIM = int(os.environ.get("MODEL_DIM", 512))` → `480`
4. All other code identical
5. Run: `RUN_ID=phase1.2_12L_leaner480 torchrun --standalone --nproc_per_node=8 train_gpt.py`

## Success Criteria
- ✓ Runs to completion in ≤10 minutes
- ✓ Model artifact ≤15.0 MB
- ✓ Loss traceable to roundtrip validation
- ✓ Beats 1.1194 (target), or ≤1.1250 acceptable for analysis
