# Phase 1.1: Int4 QAT with Adaptive Warmdown Schedule

**Challenge**: Minimize FineWeb validation loss within 16MB artifact + 10-minute training on 8×H100s

## Hypothesis

Aggressive 4-bit quantization (int4, 15 levels -8 to +7) with adaptive warmdown schedule may improve over int6 baseline by:
1. **More aggressive compression** (int4 vs int6 saves ~20-30% compared to baseline)
2. **Adaptive QAT schedule** that smoothly reduces quantization noise as training progresses
3. **Momentum preservation** via EMA (0.997) to stabilize convergence

Expected improvement: **≤1.1180-1.1200** loss (vs best record 1.1194)

## Implementation Details

### Architecture (Unchanged from Best Record 1.1271)
- **11 layers** encoder-decoder split (6+5 with skip residuals)
- **512 hidden dimension** (4 KV heads, tied embeddings)
- **Features**: 4-layer XSA (last layers), SmearGate, BigramHash(4096), NTK RoPE, logit_softcap=30
- **Quantization baseline**: int8 mixed (controls) + int6 STE QAT (weights/embeddings)

### Phase 1.1 Modifications

#### Quantization: Int4 QAT (4-bit, 15 levels)
```python
QAT_INT_BITS = 4
QAT_MAX_VALUE = 7          # Range -8 to +7 (15 levels)
QAT_MIN_VALUE = -8
```

#### Adaptive Warmdown Schedule
- **QAT start**: Step 1000 (allow optimizer to stabilize before quantization)
- **Warmdown phase**: Steps 4000-6500
- **Scale factor interpolation**: High=0.8 → Low=0.6 (linear)
  - Higher scale factor early = less aggressive clipping (allows gradients to flow)
  - Lower scale factor late = more aggressive compression (approaching final artifact precision)

#### Integration
```python
# In training loop:
def _qat_scale_factor(step):
    if step < QAT_START_STEP or step > QAT_WARMDOWN_END:
        return 0.6  # Static low for fine-grained compression
    if step < QAT_WARMDOWN_START:
        return 0.8  # Static high during early QAT
    progress = (step - QAT_WARMDOWN_START) / (QAT_WARMDOWN_END - QAT_WARMDOWN_START)
    return 0.8 - 0.2 * progress  # Linear interpolation 0.8→0.6

CastedLinear._qat_scale_factor = _qat_scale_factor(current_step)
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| **Batch** | 786,432 tokens/step, seq_len=2048 |
| **Duration** | ~10-minute wallclock cap on 8×H100s |
| **Warmup** | 20 steps (linear) |
| **Warmdown** | 1200 iters |
| **Learning Rates** | Matrix 0.04, Scalar 0.04, Tied Embed 0.05 |
| **Optimizers** | Muon (0.95 momentum) + Adam split |
| **EMA** | Enabled, decay=0.997 |
| **Quantization** | int4 STE QAT (MLP, attention) + int8 (embeddings) |
| **Compression** | zstd-22 |

## Expected Performance

| Metric | Best Case | Target | Acceptable |
|--------|-----------|--------|-----------|
| **val_bpb** | ≤1.1180 | 1.1180-1.1200 | ≤1.1220 |
| **Time** | ~9.5 min | ~9.8 min | ≤10 min |
| **Artifact** | 14.8 MB | 14.9-15.0 MB | ≤15.8 MB |

## Risk Assessment

**Risks**:
- Int4 quantization may incur >0.005 roundtrip loss (artifact precision penalty)
- Warmdown schedule complexity adds tuning surface; may need iteration
- QAT timing (step 1000-6500) may not align with optimal loss convergence

**Mitigation**:
- Baseline: Proven int6 setup (Phase 1.2) serves as fallback
- Conservative QAT start (step 1000) allows pre-training stability
- EMA maintains model quality despite quantization noise

## Reproducibility

**Seed**: 1337 (fixed)
**Training time**: Measured wallclock ms (not step count)
**Validation**: Sliding window eval + standard eval both logged
**Roundtrip**: Quantized model roundtrip loss verified post-training

## Author Notes

This variant tests whether aggressive quantization + adaptive schedule can improve the proven int6 baseline. Success hinges on:
1. QAT schedule not disrupting convergence
2. int4 compression reaching artifact size targets
3. Roundtrip loss penalty ≤0.01 nats

If Phase 1.1 underperforms, architecture (Phase 1.2) or training recipe (Phase 1.3) variants are ready as backups.
