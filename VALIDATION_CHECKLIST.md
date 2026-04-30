# Pre-Submission Checklist & Validation Guide

## ✅ Pre-Training Validation (Do This First!)

### Code Quality
- [ ] Run: `python -c "from train_gpt import Hyperparameters; Hyperparameters.log_config()"`
- [ ] Verify output shows: `schedule_mode: cosine`, `use_ema: True`
- [ ] No import errors
- [ ] Check file size: Should be ~600 lines

### Environment Setup
- [ ] Python 3.8+: `python --version`
- [ ] PyTorch with CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Required packages: `numpy`, `torch`, `torch.nn.functional`
- [ ] Venv activated: `which python` shows `.venv` path
- [ ] Data exists: `ls ./data/datasets/fineweb10B_sp1024/`
- [ ] GPU memory: `nvidia-smi` shows available VRAM

### Configuration
- [ ] Default ITERATIONS: 20000
- [ ] Default WARMUP_STEPS: 1500
- [ ] Default SCHEDULE_MODE: "cosine"
- [ ] Default USE_EMA: "true"
- [ ] Default MATRIX_LR: 0.045
- [ ] Default WEIGHT_DECAY: 0.001

---

## 🔄 Training Validation (During Training)

### Step 0 (Initialization)
- [ ] Model prints parameter count: ~26.8M
- [ ] Device shows: `cuda` (or `cpu` if no GPU)
- [ ] Config logs with all hyperparameters
- [ ] Output directory created: `./checkpoints_v4/`

### Step 100 (Early Training)
- [ ] Training loss decreases (should be < 3.0)
- [ ] Learning rate applying (check logged LR values)
- [ ] Validation runs at correct frequency
- [ ] BPB computed and logged
- [ ] No OOM errors or NaNs

### Step 1000 (After Warmup)
- [ ] Loss should be ~1.2-1.3
- [ ] Learning rate in cosine decay phase
- [ ] EMA accumulating (check shadow dict size in memory)
- [ ] Validation BPB improving gradually
- [ ] Checkpoints being saved

### Mid-Training (50% of iterations)
- [ ] Loss smooth, no sudden spikes
- [ ] Validation BPB: ~1.055-1.065
- [ ] LR decreasing toward minimum
- [ ] Training stable, no divergence
- [ ] GPU memory stable (no leaks)

### Late Training (90% complete)
- [ ] Loss: ~1.045-1.050
- [ ] Validation BPB: ~1.048-1.055
- [ ] LR in warmdown phase (very small)
- [ ] Best checkpoint saved
- [ ] EMA state maintained

### Training Complete
- [ ] Final message: "Training complete!"
- [ ] Best BPB reported: Should be < 1.055
- [ ] EMA applied and evaluated
- [ ] Final files in `./checkpoints_v4/`:
  - `best_model.pt`
  - `best_ema_model.pt`

---

## 🎯 BPB Validation

### Expected BPB Ranges by Step

| Step | v3 Baseline | v4 Expected | Status |
|------|------------|-------------|--------|
| 100 | 2.8-3.0 | 2.8-3.0 | Training |
| 1000 | 1.4 | 1.35 | Warmup done |
| 5000 | 1.100 | 1.075 | Good progress |
| 10000 | 1.085 | 1.050 | Getting close |
| 15000 | 1.075 | 1.048 | Fine-tuning |
| 20000 | 1.070 | 1.048-1.050 | **Target** ✅ |

### Sanity Checks
- [ ] BPB should monotonically decrease (generally)
- [ ] Final BPB < 1.055 (else something wrong)
- [ ] BPB with EMA < standard BPB
- [ ] Each checkpoint better than previous

### Red Flags (STOP and Debug)
```
❌ BPB = 3.0 after 100 steps → Wrong data / initialization
❌ BPB = 1.5 after 5000 steps → LR too low, adjust
❌ BPB NaN/Inf → Gradient explosion, lower LR
❌ BPB = 1.2 after 20000 steps → Very wrong, check settings
❌ BPB increasing → Overtraining, enable SWA earlier
```

---

## 📊 Performance Validation

### Training Speed
- [ ] Tokens/sec reported in logs
- [ ] Expected: 50-100k tokens/sec on A100
- [ ] Should be consistent ±10%
- [ ] No sudden slowdowns

### Memory Usage
- [ ] Monitor with: `nvidia-smi -l 1`
- [ ] Model: ~110 MB (26.8M params)
- [ ] Activations: ~500 MB typical
- [ ] EMA shadow: +110 MB
- [ ] Total: ~700-900 MB typical
- [ ] Should stay constant (no leaks)

### Loss Curve Shape
```
Expected shape:

Loss
  │
4 │ ╱                           (noise expected)
3 │╱
2 │╱
1 │╱╲╱╱╲
  │  ╲╲╱╲ (smooth after 1k steps)
0 │───────────────── Steps
  0    5k   10k   15k   20k

Key: Steep drop early, then smooth decay.
```

---

## 🏆 Leaderboard-Ready Validation

### Final Model Selection
1. **Check which model is best**:
   ```powershell
   Get-Item ./checkpoints_v4/best_*_model.pt | 
   ForEach-Object { $_.Name, (Get-Date $_.LastWriteTime) }
   ```

2. **Load and verify model size**:
   ```powershell
   python -c "
   import torch
   ckpt = torch.load('./checkpoints_v4/best_ema_model.pt')
   print(f'Keys: {list(ckpt.keys())}')
   print(f'Model params: {ckpt[\"model\"].keys()}')
   "
   ```

3. **Best candidate**:
   - Usually: `best_ema_model.pt` (if BPB improved)
   - Fallback: `best_model.pt` (standard)

### Pre-Submission Tests
- [ ] Load checkpoint without errors
- [ ] Model inference works
- [ ] Output shape correct
- [ ] No NaN/Inf in outputs
- [ ] Parameter count matches (26.8M)

---

## 📋 Training Report Template

### For Your Records
```
═══════════════════════════════════════════════════════════════
PARAMETER GOLF v4 TRAINING REPORT
═══════════════════════════════════════════════════════════════

Date:              [TODAY]
GPU:               [nvidia-smi info]
Duration:          [HH:MM]
Total Iterations:  [20000]
Final Steps:       [20000]

CONFIGURATION:
  SCHEDULE_MODE:   cosine
  USE_EMA:         true
  EMA_DECAY:       0.9995
  MATRIX_LR:       0.045
  WEIGHT_DECAY:    0.001
  WARMUP_STEPS:    1500
  WARMDOWN_STEPS:  1000

RESULTS:
  Initial Loss:    [3.0]
  Final Loss:      [X.XXX]
  Best BPB:        [1.0XX] ← Key metric
  BPB Step:        [XXXXX] (when achieved)
  
  With EMA:
    EMA BPB:       [1.0XX] ← Usually better
    Improvement:   [+0.00X]

BEST CHECKPOINT:
  File:            best_ema_model.pt
  Size:            ~107 MB
  Location:        ./checkpoints_v4/

STATUS:
  ✅ Training complete
  ✅ BPB < 1.055
  ✅ Ready for submission

LEADERBOARD PROJECTION:
  BPB:    1.0XX
  Est.:   Top 5-20 (depending on competition)
═══════════════════════════════════════════════════════════════
```

---

## 🚨 Troubleshooting Guide

### Issue: BPB Not Improving After 1000 Steps
**Diagnosis**:
- [ ] Check schedule: Is LR being applied?
- [ ] Print sample: `print(f"LR: {opt.param_groups[0]['lr']}")`
- [ ] Check data loading: Is it deterministic?

**Fix**:
- [ ] Increase `WARMUP_STEPS` to 2000
- [ ] Reduce `MATRIX_LR` to 0.040
- [ ] Verify `SCHEDULE_MODE="cosine"` is set

### Issue: Training Diverges (BPB → NaN)
**Diagnosis**:
- [ ] Gradient explosion: Check loss early (step 10-100)
- [ ] Bad hyperparameters: LR too high?
- [ ] Bad data: Check file integrity

**Fix**:
- [ ] Lower `MATRIX_LR` to 0.035
- [ ] Increase `GRAD_CLIP` to 1.5
- [ ] Increase `WARMUP_STEPS` to 2000

### Issue: BPB Stuck at 1.065+
**Diagnosis**:
- [ ] Training not reaching refining phase
- [ ] Warmdown too aggressive
- [ ] Settings not optimized for your GPU/data

**Fix**:
- [ ] Lower `WARMDOWN_STEPS` to 500
- [ ] Increase `WEIGHT_DECAY` to 0.002
- [ ] Run tuning pass (2k steps) first

### Issue: GPU OOM
**Diagnosis**:
- [ ] Model too large: 26.8M is fine
- [ ] EMA + SWA: Both on at once
- [ ] Sequence length: Too long

**Fix**:
- [ ] Set `USE_EMA="true"` and `USE_SWA="false"` (not both)
- [ ] Reduce `TRAIN_SEQ_LEN` to 512
- [ ] In code, reduce validation steps: `steps=10`

---

## 📋 Pre-Leaderboard Checklist

### 48 Hours Before Submission
- [ ] Run full 20k training
- [ ] Monitor complete trajectory
- [ ] Log final BPB and timestamp
- [ ] Verify EMA checkpoint improved BPB
- [ ] Save final report

### 24 Hours Before Submission
- [ ] Load checkpoint and test inference
- [ ] Verify parameter count (26.8M)
- [ ] Check file integrity
- [ ] Copy to safe location (backup)

### Day of Submission
- [ ] Run final smoke test (100 steps)
- [ ] Confirm code still works
- [ ] Load final checkpoint one more time
- [ ] Ready to submit best_ema_model.pt

---

## ✨ Success Indicators

### Your v4 is successful if:
✅ Code runs without errors  
✅ Training curve smooth and decreasing  
✅ Final BPB < 1.055  
✅ EMA checkpoint improves BPB  
✅ All checkpoints saved properly  
✅ Training completes in ~24 hours  
✅ Loss doesn't diverge  
✅ Memory stable (no leaks)  

### You're ready for leaderboard if:
✅ BPB between 1.045-1.055  
✅ Training stable and reproducible  
✅ Checkpoint loads without errors  
✅ Model inference works  
✅ File size ~107 MB  

---

## 🎯 Final Sanity Check

Run this before submission:
```powershell
# 1. Check code
python -c "from train_gpt import Hyperparameters; print('✓ Code OK')"

# 2. Check checkpoint
python -c "
import torch
ckpt = torch.load('./checkpoints_v4/best_ema_model.pt')
print(f'✓ Checkpoint loads')
print(f'  BPB: {ckpt[\"bpb\"]:.4f}')
print(f'  Step: {ckpt[\"step\"]}')
"

# 3. Check size
ls -lh ./checkpoints_v4/best_ema_model.pt

# 4. You're good to go!
```

---

**Ready?** → Upload `best_ema_model.pt` to leaderboard! 🚀
