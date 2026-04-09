# Legal Techniques Only — Parameter Golf Strategy

**Decision Date**: 2026-04-09  
**Reason**: Pre-quant TTT likely violates challenge rules (trains on full val set before any scoring)

---

## The Rules (Clear Interpretation)

**Track A (No Adaptation):**
- Train on training data only
- No exposure to validation data before evaluation
- Quantization, architecture, hyperparameters all fair game

**Track B (Score-First Adaptation):**
- Must evaluate tokens FIRST (get loss)
- Then adapt on already-scored tokens only
- Apply adaptation to future tokens (causal, left-to-right)
- One pass through evaluation data

**Pre-Quant TTT Violates Both:**
- ✗ Not Track A: Sees all val tokens across 6-10 epochs before any scoring
- ✗ Not Track B: Not causal, not score-first, not one-pass

---

## Legal Techniques We Can Use

### 1. Architecture Improvements (Highest Priority)

| Technique | Source | Expected Impact |
|-----------|--------|-----------------|
| **Depth Recurrence** | PR #1471, #1487 | ~0.01-0.02 BPB (vs our looping) |
| **Parallel Residuals** | Our Run 007/008 | ~0.003-0.005 BPB |
| **Looping** | Our Run 007/008 | ~0.005-0.01 BPB |
| **QK-Gain Tuning** | PR #1487 (5.25) | ~0.001-0.002 BPB |
| **EMA Decay** | Literature (0.9965) | ~0.0005-0.001 BPB |
| **Skip Gates** | PR #1471 | ~0.002-0.003 BPB |

**Action**: Test depth recurrence (layers 3,4,5) vs our current looping (layers 4,5). This is the biggest unknown — PR #1487 uses 3-layer depth recurrence, we use 2-loop on 2 layers.

### 2. Quantization Improvements

| Technique | Source | Expected Impact |
|-----------|--------|-----------------|
| **SDClip** | PR #1394, #1471 | Better rate-distortion, zero pruning |
| **GPTQ int6** | Standard | Baseline |
| **Brotli Compression** | Standard | ~1-2% size reduction |
| **Int8 Embeddings** | Standard | Baseline |

**Action**: Our Run 007/008 already uses GPTQ int6 + Brotli. Could test SDClip (k·std clipping) for better quantization quality.

### 3. Tokenizer Experiments

| Approach | Vocab Size | Status |
|----------|------------|--------|
| **SP1024** (ours) | 1024 | Novel, saves ~4M params |
| **SP8192** (theirs) | 8192 | Standard, used by top submissions |

**Unknown**: Direct comparison between SP1024 and SP8192 with same architecture. Our SP1024 saves params but may lose per-token expressivity.

**Action**: Test SP8192 with our architecture (depth recurrence + looping) to isolate tokenizer effect.

### 4. Hyperparameter Tuning (Training Data Only)

| Parameter | Our Current | To Sweep |
|-----------|-------------|----------|
| Weight Decay | 0.085 | 0.090, 0.095, 0.10 |
| Matrix LR | 0.04 | 0.022, 0.03, 0.04 |
| Warmdown Frac | 0.667 | 0.72, 0.75 |
| Muon Momentum | 0.99 | 0.995 |
| QK-Gain | 5.0 | 5.25, 5.5 |

**Action**: Small sweeps on training data (not validation) to find optimal config.

### 5. Track B (Score-First TTT) — If We Want Adaptation

**Legal Implementation:**
```python
# For each sliding window:
# 1. Evaluate all tokens (get loss, no grad)
# 2. Adapt on context tokens ONLY (already scored)
# 3. Apply delta to new tokens
# 4. Move to next window (causal, one-pass)
```

**Expected Impact**: ~0.01-0.02 BPB (based on PR #1306, #1322 before SLOT concerns)

**Caveat**: Adds eval time, may not fit 10-min window

---

## Immediate Next Runs (No TTT)

### Run 010: Depth Recurrence Test

**Hypothesis**: 3-layer depth recurrence (layers 3,4,5) beats our 2-loop on layers 4,5

| Parameter | Run 007/008 | Run 010 |
|-----------|-------------|---------|
| Recurrence Type | 2-loop on L4-5 | **Depth recurrence L3-5** |
| Virtual Layers | ~13 | **14** (11 + 3) |
| TTT | 6ep pre-quant | **NONE** |
| QK-Gain | 5.0 | 5.0 |
| SP1024 | Yes | Yes |

**Expected**: If depth recurrence > looping, we gain ~0.005-0.01 BPB

### Run 011: QK-Gain + WD Sweep

**Hypothesis**: PR #1487's QK=5.25 + higher WD improves our baseline

| Parameter | Run 007/008 | Run 011 |
|-----------|-------------|---------|
| QK-Gain | 5.0 | **5.25** |
| Weight Decay | 0.085 | **0.095** |
| TTT | 6ep pre-quant | **NONE** |
| Recurrence | 2-loop L4-5 | 2-loop L4-5 |

**Expected**: ~0.002-0.003 BPB from hyperparameter tuning alone

### Run 012: SP8192 Comparison

**Hypothesis**: SP8192 with our architecture beats SP1024

| Parameter | Run 007/008 | Run 012 |
|-----------|-------------|---------|
| Tokenizer | SP1024 | **SP8192** |
| Recurrence | 2-loop L4-5 | 2-loop L4-5 |
| TTT | 6ep pre-quant | **NONE** |
| QK-Gain | 5.0 | 5.0 |

**Expected**: Isolates tokenizer effect; if SP8192 wins, we know param savings aren't worth it

---

## Competitive Position (Post-TTT Pivot)

| Submission | BPB | Uses Pre-Quant TTT? | Legality Risk |
|------------|-----|---------------------|---------------|
| PR #1488 | 0.8265 | SLOT (different issue) | HIGH (SLOT legality) |
| PR #1487 | 1.0600 | **Yes** | **MEDIUM-HIGH** |
| PR #1485 | 1.0679 | **Yes** | **MEDIUM-HIGH** |
| **Our Run 007/008** | **1.07389** | **Yes** | **MEDIUM-HIGH** |
| PR #1019 (Official SOTA) | 1.1147 | No | LOW (merged) |

**If TTT is ruled illegal:**
- Top 3 submissions (#1487, #1485, our Run 007/008) all disqualified
- Official SOTA reverts to PR #1019 (1.1147 BPB)
- We need a **legal** submission beating 1.1147

**If TTT is ruled legal:**
- We're at 1.07389, ~0.014 BPB behind #1487
- Need ~0.014 BPB from architecture + hyperparameter improvements

**Our Edge**: Clean submission, no controversial techniques (SLOT), reproducible data

---

## Summary

**Stop Immediately:**
- ✗ Pre-quant TTT (any variant that sees val tokens before scoring)

**Continue/Pivot To:**
- ✓ Architecture improvements (depth recurrence, looping, parallel residuals)
- ✓ Quantization (GPTQ, SDClip, Brotli)
- ✓ Hyperparameter tuning on **training** data
- ✓ Tokenizer experiments (SP1024 vs SP8192)
- ✓ Track B score-first TTT (if we want adaptation, must be causal)

**Next Run**: Run 010 — Depth recurrence test (no TTT)
