# Experimental Findings — Parameter Golf (April 2026)

## Objective
Beat PR #1493's SOTA of **1.0810 BPB** (3-seed mean) in the parameter-golf competition.  
Constraints: 16,000,000 bytes (decimal) artifact cap, 600s train+eval wallclock, 8×H100 GPUs.

## Baseline: PR #1493
- **Architecture**: 11L×512d×8H/4KV, 3-layer recurrence (L3-5 at frac=0.35), parallel residuals (L7+), tied embeddings, logit softcap=30, XSA on all 11 layers, U-Net skip connections
- **Quantization**: int6 matrices (clip_sigmas=12.85, clip_range=31), int8 embeddings (clip_sigmas=20.0, clip_range=127), brotli compression, GPTQ
- **Results** (SEED=42, QK_GAIN_INIT=5.25):
  - Pre-quant non-sliding: **1.08757**
  - Post-quant non-sliding: **1.10014** (quant gap = 0.01257)
  - Post-quant sliding: **1.08329**
  - Post-quant TTT: **1.08103**

---

## Experiment 1: Quantization-Aware Training (QAT)

**Hypothesis**: Fake-quantize STE during warmdown lets the model adapt to quantization noise, reducing the quant gap.

**Implementation** (`train_qat.py`):
- Forward: `w + (quantize(w) - w).detach()` — uses quantized weights, backward passes gradients as identity
- Applied to all CastedLinear (numel > 65536) + embedding when `_QAT_ACTIVE=True`
- Toggled on at configurable training fraction

**Runs**:
| Run | Start Frac | Strategy | Post-Quant Non-Sliding | Notes |
|-----|-----------|----------|----------------------|-------|
| run1 | 0.28 | Standard | CRASHED | `FailOnRecompileLimitHit` (fixed with `cache_size_limit=32`) |
| run2 | 0.28 | Standard | 1.1118 | +0.012 worse than baseline |
| run3 | 0.55 | Standard | 1.1111 | Similar degradation |
| run4 | 0.85 | Standard | 1.1092 | Slightly better but still worse |
| run5 | 0.28 | EMA-then-finetune | 1.1194 | Worst — optimizer state mismatch |

**Root Cause**: EMA contamination. With decay=0.9965, the last ~300 steps dominate the EMA average. QAT noise in those final steps contaminates the EMA irreversibly. The EMA model (which gets saved) never fully benefits from QAT adaptation.

**Conclusion**: QAT is incompatible with this training setup. The EMA mechanism, which is critical for final model quality, cannot coexist with QAT noise injection.

---

## Experiment 2: Partial Key Offset (PKO)

**Hypothesis**: Shifting non-RoPE key dimensions by 1 position enables implicit bigram/position awareness.

**Implementation** (`train_pko.py`, `eval_pko.py`):
```python
rd = self.rope_dims  # 16
if seqlen > 1:
    k = k.clone()
    k[:, 1:, :, rd:] = k[:, :-1, :, rd:]
```

**Results**:

### Eval-only PKO (applied to pre-trained model without retraining):
- Baseline: **1.10014**
- PKO all layers: **~1.68** (catastrophic failure)
- PKO encoder-only: **~1.68** (catastrophic failure)

### Training with PKO:
- Pre-quant: **1.08829** (vs baseline 1.08757, +0.0007 worse)
- Post-quant sliding: **1.08420** (vs baseline 1.08329, +0.0009 worse)
- **Post-quant TTT: 1.10474** (catastrophic — TTT makes it WORSE by 0.02 BPB)

**Root Cause**: PKO shifts create non-standard key representations that break TTT's gradient-based weight adaptation. TTT assumes standard attention semantics; the shifted keys create an optimization landscape that gradient updates can't navigate.

**Conclusion**: PKO is incompatible with TTT. Since TTT provides the critical final ~0.002 BPB gain, PKO is a net negative.

---

## Experiment 3: Mixed-Precision Sensitivity Scan

**Hypothesis**: With code packing (~16,600 bytes for submission code), we have freed budget to promote high-sensitivity layers from int6 to int7.

**Implementation** (`sensitivity_scan.py`):
- For each quantizable matrix: test int7 (clip_range=63) while keeping others at int6
- Measure extra compressed bytes and BPB improvement

**Results**:
- Estimated budget: 6,447 bytes (16M - baseline_model - 16,600 code)
- Minimum cost to promote any matrix: ~16,500 bytes (even the smallest 131K-param layers)
- **ALL matrices marked OVER budget**

**Conclusion**: The byte budget is far too tight for mixed precision. Even the smallest matrix promotion exceeds available margin after compression.

---

## Strategic Analysis: What's Left on the Table

### High-EV opportunities (not yet tested):
1. **MTP (Multi-Token Prediction)** — auxiliary loss predicting t+2, t+3 tokens during training. Heads discarded at inference → zero byte cost. v13 codebase already has an implementation. Estimated potential: 0.002-0.005 BPB pre-quant improvement.
2. **More recurrence at eval-only** — using num_loops=3 during eval (trained with 2). May give ~0.001 BPB gain at inference time cost within 600s budget.
3. **Hyperparameter tuning** — warmdown_frac, learning rate schedule, MuonEQR parameters.

### Dead ends (definitively ruled out):
- QAT with EMA-based training
- PKO (incompatible with TTT)
- Mixed precision promotion (budget too tight)
- Eval-only architectural modifications on pre-trained models

### Key insight:
The quant gap (0.0126 BPB) is the biggest single loss. Reducing it requires either:
- Better GPTQ (different Hessian estimation, more calibration data)
- Smaller model that quantizes better (but then pre-quant suffers)
- Training changes that make weights more quantization-friendly WITHOUT explicit QAT noise

---

## File Index

| File | Purpose |
|------|---------|
| `train_qat.py` | QAT-modified training script |
| `train_pko.py` | PKO-modified training script |
| `eval_pko.py` | Eval-only PKO test |
| `sensitivity_scan.py` | Mixed-precision sensitivity scanner |
| `run_qat.sh` | QAT run configuration |
| `logs/qat_run*.log` | QAT training logs (5 runs) |
| `logs/pko_run1.log` | PKO training log |
| `logs/eval_pko.log` | PKO eval-only log |
| `logs/sensitivity_scan.log` | Sensitivity scan results |
| `logs/baseline_restore.log` | Baseline restoration log |
