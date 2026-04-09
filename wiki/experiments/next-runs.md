# Parameter Golf — Future Run Ideas

**Last Updated**: 2026-04-09  
**Constraint**: NO pre-quant TTT (illegal — trains on val before scoring). Track B score-first causal TTT is legal.

---

## Run Queue (Priority Order)

### Run 010: Track A Baseline — Depth Recurrence Test

**Priority**: HIGH (establish legal baseline)  
**Hypothesis**: 3-layer depth recurrence (L3-5) beats our 2-loop on L4-5  
**Status**: **READY TO SUBMIT**

| Parameter | Run 007/008 | Run 010 |
|-----------|-------------|---------|
| Recurrence | 2-loop on L4-5 | **Depth recurrence L3-5** |
| TTT | 6ep pre-quant (illegal) | **NONE** |
| QK-Gain | 5.0 | 5.25 |
| Weight Decay | 0.085 | 0.095 |
| Tokenizer | SP1024 | SP1024 |

**Expected**: ~1.08-1.09 BPB (architecture gain offsets TTT loss)

**Files**:
- `records/track_10min_16mb/2026-04-09_SP1024_Recur345_NoTTT/`
- `train_gpt.py` (depth recurrence implementation)
- `run_all_seeds.sh` (3 seeds: 314, 42, 999)
- `README.md`, `submission.json`

---

### Run 011: Track A — Hyperparameter Sweep

**Priority**: MEDIUM  
**Hypothesis**: Training-data-only tuning gains ~0.003-0.005 BPB

| Parameter | Run 010 | Run 011 Sweep |
|-----------|---------|---------------|
| Weight Decay | 0.095 | 0.090, 0.095, 0.10 |
| Matrix LR | 0.04 | 0.022, 0.03, 0.04 |
| Warmdown Frac | 0.667 | 0.72, 0.75 |
| QK-Gain | 5.25 | 5.0, 5.25, 5.5 |

**Expected**: Best combo ~1.075-1.085 BPB  
**Status**: After Run 010 results

---

### Run 012: Track A — SP8192 Comparison

**Priority**: MEDIUM  
**Hypothesis**: SP8192 with our architecture beats SP1024

| Parameter | Run 010 | Run 012 |
|-----------|---------|---------|
| Tokenizer | SP1024 | **SP8192** |
| Recurrence | Depth L3-5 | Depth L3-5 |
| TTT | None | None |

**Expected**: Isolates tokenizer effect; if SP8192 wins by >0.005 BPB, switch  
**Status**: After Run 010/011

---

### Run 013: Track B — Score-First TTT (Causal)

**Priority**: HIGH (competitive ceiling)  
**Hypothesis**: Legal Track B TTT gains ~0.01-0.015 BPB over Track A

**Implementation** (per-window causal):
```python
for window in sliding_windows(val_data):
    # Step 1: Score all tokens (no grad)
    losses = evaluate_window(model, window)
    
    # Step 2: Adapt on CONTEXT tokens only (already scored)
    context_tokens = window[:-64]  # All but last stride
    delta = adapt_on_tokens(model, context_tokens, epochs=1)
    
    # Step 3: Apply delta to score NEW tokens
    new_tokens = window[-64:]  # Last stride (unscored)
    losses_new = evaluate_window(model + delta, new_tokens)
```

| Parameter | Run 010 | Run 013 |
|-----------|---------|---------|
| TTT Type | None | **Track B score-first** |
| TTT Epochs | N/A | 1 (causal, per-window) |
| Adaptation Scope | N/A | Context tokens only |

**Expected**: ~1.06-1.07 BPB (beats PR #1487 if architecture is strong)  
**Risk**: Adds eval time — must fit 10-min window  
**Status**: After Track A baseline established

---

### Run 014: Track A — SDClip Quantization

**Priority**: LOW (incremental gain)  
**Hypothesis**: SDClip (k·std clipping) beats percentile search

| Parameter | Run 010 | Run 014 |
|-----------|---------|---------|
| Quantization | GPTQ int6 | **SDClip int6** |
| Clip Method | Multi-percentile | **k · std(row)** |
| k (matrices) | N/A | 12.85 |
| k (embeddings) | N/A | 20.0 |

**Source**: PR #1394, #1471 (zero selective pruning)  
**Expected**: ~0.001-0.002 BPB, better rate-distortion  
**Status**: If we need marginal gains

---

### Run 015: Track A — Combined Architecture

**Priority**: MEDIUM (kitchen sink)  
**Hypothesis**: Best legal techniques compound

| Component | Source | Expected Contribution |
|-----------|--------|----------------------|
| Depth Recurrence L3-5 | PR #1487 | ~0.005-0.01 BPB |
| Parallel Residuals L7+ | Run 007/008 | ~0.003 BPB |
| QK-Gain 5.25 | PR #1487 | ~0.001-0.002 BPB |
| EMA 0.9965 | Literature | ~0.0005-0.001 BPB |
| WD 0.095 | PR #1331 | ~0.001-0.002 BPB |
| Warmdown 0.72 | PR #1445 | ~0.001 BPB |
| SP1024 or SP8192 | TBD | Baseline |

**Expected**: ~1.065-1.075 BPB (Track A ceiling)  
**Status**: After individual components validated

---

## Competitive Targets

| Submission | BPB | Technique | Legality |
|------------|-----|-----------|----------|
| PR #1488 | 0.8265 | SLOT-24 | HIGH risk (SLOT illegal?) |
| PR #1487 | 1.0600 | Pre-quant TTT | MEDIUM-HIGH (TTT illegal) |
| PR #1485 | 1.0679 | Pre-quant TTT | MEDIUM-HIGH (TTT illegal) |
| **Our Run 007/008** | 1.07389 | Pre-quant TTT | MEDIUM-HIGH (TTT illegal) |
| PR #1019 | 1.1147 | No TTT | LOW (official merged SOTA) |

**If TTT ruled illegal**: Beat 1.1147 with Track A → easy win  
**If TTT ruled legal**: Beat 1.0600 with Track B + architecture → harder but possible

---

## Technique Notes

### Depth Recurrence vs. Looping

| Aspect | Depth Recurrence | Looping |
|--------|-----------------|---------|
| Implementation | Reuse weights within forward pass | Iterate over layers multiple times |
| Virtual Layers | 11 + 3 = 14 | 11 × 2 = 22 (but shared weights) |
| Memory | Lower (no extra activations) | Higher (stores intermediate) |
| Source | PR #1471, #1487 | Our Run 007/008 |
| Direct Comparison | **Unknown** — needs Run 010 |

### Track B TTT Implementation Notes

**Legal Pattern** (from Issue #1336):
1. Evaluate token t → get loss → lock score
2. Add token t to "already scored" set
3. Adapt model on "already scored" set
4. Use adapted model for tokens t+1, t+2, ...
5. Never adapt on unscored tokens

**Per-Window Pattern** (sliding window eval):
- Window: 2048 tokens, stride 64
- Context: tokens 0-1983 (scored in prior windows)
- New: tokens 1984-2047 (to be scored now)
- Adapt on context → apply to new → score new → slide

**Time Budget**:
- Training: ~580s
- Track B TTT: adds ~20-40s (1 epoch per window)
- Total target: <600s

---

## Decision Log

**2026-04-09**: Discovered pre-quant TTT is illegal. Cancelled Run 009. Pivot to Track A baseline first, then Track B score-first TTT.

**Key Insight**: The maintainers confirmed TTT is legal ONLY when causal/score-first. Pre-quant TTT violates this by seeing all val tokens before any scoring.
