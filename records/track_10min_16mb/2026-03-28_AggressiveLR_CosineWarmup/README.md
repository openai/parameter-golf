# Phase 1.3: Aggressive LR Schedule with Cosine Warmup

## Overview
Optimizes training recipe alone (no architecture/quantization changes) to maximize convergence in 10-minute window with longer linear warmup and cooler LR tail.

## Hypothesis
- **Longer warmup** (20→2000 steps) primes the loss landscape better, leading to faster convergence
- **Cooler final tail** (end_lr_ratio: 0.1→0.05) allows fine-grained final optimization
- **EMA decay 0.999** (vs 0.997) creates smoother weight averaging, better final checkpoint
- **Explicit warmdown** last 500 steps before eval locks in gains
- Target: Loss ≤1.1200 with better stability across runs

## Key Configuration Changes
```python
# From best record (1.1271):
WARMUP_STEPS: 20 → 2000  (100x longer linear warmup)
WARMDOWN_ITERS: 1200 → 2000 (explicit spend more time cooling)
EMA_DECAY: 0.997 → 0.999  (heavier averaging)
END_LR_RATIO: (new) 0.05 (cooler cosine tail)

# Model architecture: UNCHANGED
# - num_layers = 11, model_dim = 512, mlp_mult = 3.0
# - int6 quantization, XSA, SmearGate, BigramHash
# - Quantization function identical
```

## LR Schedule Design
```python
def lr_mul_cw(step: int, elapsed_ms: float, total_steps: int = 15000) -> float:
    """Cosine annealing with explicit warmup and cooldown phases."""
    # Phase 1: Linear warmup (0 → 2000 steps)
    if step < 2000:
        return step / 2000.0  # Ramp from 0 to 1
    
    # Phase 2: Cosine annealing (2000 → 12500 steps)
    elif step < 12500:
        progress = (step - 2000) / (12500 - 2000)
        end_lr_ratio = 0.05  # Cool tail
        return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - end_lr_ratio) + end_lr_ratio
    
    # Phase 3: Final cooldown (12500 → 13000 steps)
    else:
        return max(0.0, (13000 - step) / 500.0)
```

## Key Hyperparameter Values
- **Warmup**: 2000 steps (128M tokens at 524k tokens/step)
- **Cosine phase**: ~10500 steps
- **Warmdown/cooldown**: 500 explicit steps
- **EMA decay**: 0.999 (0.2% update per step vs 0.3% before)
- **Matrix LR**: 0.04 (unchanged, Muon auto-scales)
- **Scalar LR**: 0.04 (unchanged)

## Difference from Best Record (1.1271)
- **Warmup duration**: 20 steps → 2000 steps (100×)
- **LR schedule**: Immediate cosine → Delayed cosine with long ramp
- **End LR ratio**: 0.1 (implicit) → 0.05 (explicit cooler)
- **EMA decay**: 0.997 → 0.999
- **Warmdown**: Implicit wallclock-based → Explicit step-based 500 steps

## Training Configuration (Identical Except LR Schedule)
- **Duration**: ~10 minutes on 8×H100 (wallclock cap enforced)
- **Batch**: 786,432 tokens/step, seq_len=2048
- **Optimizer**: Muon + Adam (same split, same base LRs)
- **XSA**: Last 4 layers
- **Quantization**: int6 STE QAT, zstd-22

## Expected Outcome
- **Best case**: ≤1.1170 if longer warmup helps convergence signal
- **Moderate**: 1.1200-1.1220 (similar to best record, reliable)
- **Realistic**: 1.1240-1.1280 (schedule changes are subtle)

## Implementation Notes
1. Base: Copy best record `train_gpt.py`
2. Modify Hyperparameters class:
   - Change `WARMUP_STEPS = 20` → `2000`
   - Change `WARMDOWN_ITERS = 1200` → `2000`
   - Change `EMA_DECAY = 0.997` → `0.999` 
   - Add `END_LR_RATIO = 0.05`
3. Modify `lr_mul()` function to implement cosine schedule with longer warmup (see design above)
4. Rest of code identical
5. Run: `RUN_ID=phase1.3_aggressive_lr torchrun --standalone --nproc_per_node=8 train_gpt.py`

## Success Criteria
- ✓ Runs to completion in ≤10 minutes
- ✓ Model artifact ≤15.5 MB
- ✓ Loss stable (std dev <0.001 across 2 runs)
- ✓ Beats 1.1194 or reproducibly matches it (≤1.1210)

## Notes on Subtle Changes
- Higher EMA decay (0.999 vs 0.997) means slower weight updates; beneficial for stability
- Longer explicit warmup allows model to escape sharp minima earlier
- Cooler LR tail (0.05 vs 0.1) trades final speed for precision
- Net effect: incremental, expect ≤0.005 improvement if effective
