# v4 Quick Reference Card

## 🚀 Start Here

### Option A: Quick Test (5 min)
```powershell
$env:ITERATIONS=500
python train_gpt.py
```

### Option B: Competitive Run (24 hours)
```powershell
$env:ITERATIONS=20000
$env:SCHEDULE_MODE="cosine"
$env:USE_EMA="true"
python train_gpt.py
```

### Option C: Max Performance (24+ hours)
```powershell
$env:ITERATIONS=20000
$env:SCHEDULE_MODE="cosine"
$env:USE_EMA="true"
$env:USE_SWA="true"
$env:SWA_START=18000
python train_gpt.py
```

---

## 🎯 Expected Results

| Config | BPB | Time | Notes |
|--------|-----|------|-------|
| v3 baseline | 1.070 | 24h | Original |
| v4 cosine only | 1.055-1.060 | 24h | Easy win |
| v4 + EMA | 1.048-1.053 | 24h | Recommended |
| v4 + EMA + SWA | 1.045-1.050 | 24h | Best results |

---

## 📊 Top 5 Key Changes

1. **Cosine Schedule** - `SCHEDULE_MODE="cosine"` → Smooth LR decay
2. **EMA** - `USE_EMA="true"` → Weight averaging (post-training)
3. **Gated Hash** - Built-in → Better embeddings
4. **Weight Decay** - `WEIGHT_DECAY=0.001` → Regularization
5. **Per-Head Scaling** - Built-in → Better attention

---

## 🔧 Most Important Env Vars

```powershell
# Optimizer
$env:MATRIX_LR=0.045          # Weight LR
$env:WEIGHT_DECAY=0.001       # L2 regularization

# Schedule
$env:SCHEDULE_MODE="cosine"   # Use cosine decay
$env:WARMUP_STEPS=1500        # Warmup length
$env:WARMDOWN_STEPS=1000      # Cool-down phase

# Checkpointing
$env:USE_EMA="true"           # Enable EMA
$env:USE_SWA="false"          # Optional SWA
$env:VAL_CHECK_FREQ=100       # Validation frequency
```

---

## ✅ Validation Checklist

- [ ] Run 2k step test
- [ ] Verify cosine schedule working (LR should decay)
- [ ] Check EMA output at end
- [ ] Compare BPB to baseline v3
- [ ] If 1.050-1.060 BPB: proceed to full run
- [ ] If < 1.050 BPB: tune LR slightly
- [ ] Run final 20k steps
- [ ] Apply best EMA/SWA checkpoint

---

## 📈 Tuning If Needed

**BPB too high (> 1.060)?**
- ↓ Reduce `MATRIX_LR` to 0.040
- ↑ Increase `WARMUP_STEPS` to 2000
- ↑ Increase `WEIGHT_DECAY` to 0.002

**BPB low but LR schedule weird?**
- Verify `SCHEDULE_MODE="cosine"` is set
- Check terminal output for LR values
- Try `WARMDOWN_STEPS=500` (less aggressive)

**Training diverging?**
- Lower `MATRIX_LR` to 0.040
- Increase `GRAD_CLIP` to 1.5
- Disable SWA temporarily

---

## 💾 Output Files

```
checkpoints_v4/
├── best_model.pt          ← Standard best model
├── best_ema_model.pt      ← EMA-averaged model (usually best)
└── best_swa_model.pt      ← SWA-averaged model (if enabled)
```

**Which to submit?** Usually `best_ema_model.pt` (if BPB improved over standard).

---

## 🐛 Debug Commands

```powershell
# See all hyperparameters
python -c "from train_gpt import Hyperparameters; Hyperparameters.log_config()"

# Quick syntax check
python -m py_compile train_gpt.py

# Test data loading
$env:ITERATIONS=1
python train_gpt.py

# Monitor GPU
while ($true) { nvidia-smi; Start-Sleep 2; }
```

---

## 📞 Common Questions

**Q: What's the difference between EMA and SWA?**  
A: EMA tracks during training. SWA averages late checkpoints. Both help; EMA is default.

**Q: Should I use XSA?**  
A: Only if exploring efficiency. Not for BPB improvements. Leave disabled.

**Q: What's depth recurrence?**  
A: Optionally reuse middle layers. Limited benefit. Leave disabled.

**Q: Can I resume training?**  
A: Not built-in. Save checkpoint path externally if needed.

**Q: GPU OOM?**  
A: Reduce `TRAIN_SEQ_LEN=512`, or reduce validation steps in code.

---

## 🎓 Understanding the Improvements

```
Training trajectory with v4:
Iterations →

Loss  │  
      │ ╲    ╲ (linear v3)
      │  ╲    ╲
      │   ╲   ╲
      │    ╲   ╲╲
      │  ┌──╲──╲─╲─╲ (cosine + EMA)
      │  │  ╲ ╲  ╲ ╲
      │  │   ╲ ╲  ╲ ╲
      │  │    ╲ ╲  ╲ ╲
      │  │     ╲ ╲  ╲ ╲
      │  │      ╲ ╲──╲─╲─→ EMA smooths final noise
      └──┴───────╲────────

Key wins:
- Cosine decay: Smoother training
- EMA: Reduces noise, better generalization
- Gated hash: Better embeddings
```

---

## 🏆 Leaderboard Projection

- **Conservative** (cosine + EMA): **1.050** → Top 20-30
- **Expected** (all tuned): **1.048** → Top 10-15
- **Optimistic** (perfect tune): **1.045** → Top 5

Your goal: **1.048 BPB** with solid tuning.

---

**For detailed guidance**: See `UPGRADE_GUIDE_V4.md`  
**For technical details**: See `TECHNICAL_DETAILS_V4.md`  
**For code review**: See `train_gpt.py` (fully commented)
