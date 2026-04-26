# Copy-Paste Run Commands (v4)

## 0️⃣ First Time Setup
```powershell
cd c:\Users\ASUS\Documents\Backend\parameter-golf
& .\.venv\Scripts\Activate.ps1
```

---

## 1️⃣ Validate Code (30 seconds)
```powershell
python -c "from train_gpt import Hyperparameters; Hyperparameters.log_config()"
```
Expected: Prints all hyperparameters. Shows `schedule_mode: cosine`, `use_ema: True`.

---

## 2️⃣ Smoke Test (5 minutes)
```powershell
$env:ITERATIONS=100
$env:WARMUP_STEPS=10
$env:SCHEDULE_MODE="cosine"
$env:USE_EMA="true"
$env:VAL_CHECK_FREQ=25
python train_gpt.py
```

Expected output:
```
=== CONFIG ===
...

Device: cuda
==================================================
Model: 26,796,544 parameters
==================================================

Starting training: 100 iterations | Warmup: 10 | Warmdown: 50

[00025] loss=X.XXXX | val_loss=X.XXXX val_bpb=X.XXXX | ...
...
Training complete!

Applying EMA checkpoint...
EMA Validation BPB: X.XXXX
```

---

## 3️⃣ Fast Tuning (2-4 hours, 2k steps)
```powershell
$env:ITERATIONS=2000
$env:WARMUP_STEPS=300
$env:WARMDOWN_STEPS=200
$env:SCHEDULE_MODE="cosine"
$env:USE_EMA="true"
$env:EMA_DECAY=0.999
$env:MATRIX_LR=0.045
$env:SCALAR_LR=0.02
$env:TIED_EMBED_LR=0.03
$env:WEIGHT_DECAY=0.001
$env:GRAD_CLIP=1.0
$env:VAL_CHECK_FREQ=100
$env:CHECKPOINT_DIR="./checkpoints_v4_tune"
python train_gpt.py
```

Expected BPB: **1.060-1.070** (if good settings)  
Adjust if needed:
- BPB too high? → Lower `MATRIX_LR=0.040`
- BPB too low? → Shorten `WARMDOWN_STEPS=100`

---

## 4️⃣ Competitive 20k Run (24 hours) ⭐ MAIN
```powershell
$env:ITERATIONS=20000
$env:WARMUP_STEPS=1500
$env:WARMDOWN_STEPS=1000
$env:SCHEDULE_MODE="cosine"
$env:USE_EMA="true"
$env:EMA_DECAY=0.9995
$env:USE_SWA="false"
$env:MATRIX_LR=0.045
$env:SCALAR_LR=0.02
$env:TIED_EMBED_LR=0.03
$env:WEIGHT_DECAY=0.001
$env:GRAD_CLIP=1.0
$env:VAL_CHECK_FREQ=200
$env:CHECKPOINT_DIR="./checkpoints_v4_final"
python train_gpt.py
```

Expected BPB: **1.048-1.055** ✅  
Output files: 
- `./checkpoints_v4_final/best_model.pt` (standard)
- `./checkpoints_v4_final/best_ema_model.pt` (recommended) ⭐

---

## 5️⃣ SWA Variant (20k + SWA accumulation)
```powershell
$env:ITERATIONS=20000
$env:WARMUP_STEPS=1500
$env:WARMDOWN_STEPS=1000
$env:SCHEDULE_MODE="cosine"
$env:USE_EMA="true"
$env:EMA_DECAY=0.9995
$env:USE_SWA="true"
$env:SWA_START=18000
$env:MATRIX_LR=0.045
$env:SCALAR_LR=0.02
$env:TIED_EMBED_LR=0.03
$env:WEIGHT_DECAY=0.001
$env:GRAD_CLIP=1.0
$env:VAL_CHECK_FREQ=200
$env:CHECKPOINT_DIR="./checkpoints_v4_swa"
python train_gpt.py
```

Expected BPB: **1.045-1.050** ✅✅  
Output:
- `./checkpoints_v4_swa/best_swa_model.pt` (check this!)
- `./checkpoints_v4_swa/best_ema_model.pt` (fallback)

---

## 6️⃣ Aggressive Tuning (if needed)
```powershell
$env:ITERATIONS=20000
$env:WARMUP_STEPS=2000
$env:WARMDOWN_STEPS=800
$env:SCHEDULE_MODE="cosine"
$env:USE_EMA="true"
$env:EMA_DECAY=0.99995
$env:MATRIX_LR=0.040
$env:WEIGHT_DECAY=0.002
$env:VAL_CHECK_FREQ=150
$env:CHECKPOINT_DIR="./checkpoints_v4_aggressive"
python train_gpt.py
```

Use if: Standard run gave BPB > 1.055

---

## 7️⃣ Conservative Tuning (if training diverges)
```powershell
$env:ITERATIONS=20000
$env:WARMUP_STEPS=2500
$env:WARMDOWN_STEPS=1500
$env:SCHEDULE_MODE="cosine"
$env:USE_EMA="true"
$env:EMA_DECAY=0.999
$env:MATRIX_LR=0.035
$env:WEIGHT_DECAY=0.0005
$env:GRAD_CLIP=1.5
$env:VAL_CHECK_FREQ=250
$env:CHECKPOINT_DIR="./checkpoints_v4_conservative"
python train_gpt.py
```

Use if: Training diverges or loss explodes

---

## 8️⃣ XSA Experimental (sparse attention)
```powershell
$env:ITERATIONS=15000
$env:WARMUP_STEPS=1200
$env:WARMDOWN_STEPS=800
$env:SCHEDULE_MODE="cosine"
$env:USE_EMA="true"
$env:USE_XSA="true"
$env:XSA_START_LAYER=7
$env:XSA_RATIO=0.5
$env:MATRIX_LR=0.045
$env:WEIGHT_DECAY=0.001
$env:VAL_CHECK_FREQ=100
$env:CHECKPOINT_DIR="./checkpoints_v4_xsa"
python train_gpt.py
```

Use if: Exploring efficiency (not recommended for BPB)

---

## 9️⃣ Minimal (v3 compatibility check)
```powershell
$env:ITERATIONS=20000
$env:SCHEDULE_MODE="linear"
$env:USE_EMA="false"
$env:MATRIX_LR=0.045
$env:VAL_CHECK_FREQ=100
python train_gpt.py
```

Use if: You want v3 behavior (baseline comparison)

---

## 🔟 Ultra-Long Run (30k steps, risky)
```powershell
$env:ITERATIONS=30000
$env:WARMUP_STEPS=2000
$env:WARMDOWN_STEPS=2000
$env:SCHEDULE_MODE="cosine"
$env:USE_EMA="true"
$env:EMA_DECAY=0.99995
$env:USE_SWA="true"
$env:SWA_START=25000
$env:MATRIX_LR=0.045
$env:WEIGHT_DECAY=0.001
$env:VAL_CHECK_FREQ=300
$env:CHECKPOINT_DIR="./checkpoints_v4_ultra"
python train_gpt.py
```

Use if: Maximum squeeze, GPU budget allows

---

## 📋 Decision Tree

```
Start here:
    ↓
Run Quick Test (Option 2)
    ↓ OK? 
    ├─→ NO: Diagnostics needed
    │
    ↓ YES
    ├→ Run Tuning (Option 3, 2k steps)
    │   ├─ BPB > 1.055? Use Aggressive (Option 6)
    │   ├─ BPB < 1.050? Proceed to Competitive
    │   └─ 1.050-1.055? Perfect, use those settings
    │
    ↓
    └→ Run Competitive (Option 4, 20k steps) ⭐
        ├─ Good BPB (< 1.052)? Submit best_ema_model.pt
        └─ Not great? Try SWA variant (Option 5)
```

---

## ✅ Pre-Run Checklist

Before each run:
- [ ] Activate venv: `& .\.venv\Scripts\Activate.ps1`
- [ ] Current dir: `c:\Users\ASUS\Documents\Backend\parameter-golf`
- [ ] Data exists: `ls .\data\datasets\fineweb10B_sp1024\`
- [ ] GPU available: `nvidia-smi`
- [ ] No other jobs: `nvidia-smi` (check processes)

---

## 📊 Expected Times

| Run | Steps | Time | BPB |
|-----|-------|------|-----|
| Smoke | 100 | 5 min | - |
| Tuning | 2k | 2-4 hours | 1.060 |
| Competitive | 20k | 20-24 hours | **1.048** |
| Ultra | 30k | 36 hours | 1.045 |

---

## 🎯 Recommended Sequence

**Day 1**: 
1. Run smoke test (5 min)
2. Run tuning variant (4 hours)
3. Analyze results

**Day 2-3**: 
4. Run competitive 20k (24 hours)
5. Check final BPB
6. If < 1.050, submit best_ema_model.pt
7. If > 1.052, retry with aggressive settings

---

## 💡 Pro Tips

- **Monitor** training with `nvidia-smi -l 1`
- **Early stop** if BPB > 1.055 after 10k steps (settings bad)
- **Save logs**: Redirect to file with ` | Tee-Object log.txt`
- **Compare runs**: Keep `checkpoints_v4_*` dirs separate
- **Best model**: Always pick `best_ema_model.pt` if available

---

## 🚀 TL;DR - Just Do This

```powershell
# Setup once
cd c:\Users\ASUS\Documents\Backend\parameter-golf
& .\.venv\Scripts\Activate.ps1

# Quick test
$env:ITERATIONS=100
python train_gpt.py

# Full run (24 hours)
$env:ITERATIONS=20000
$env:SCHEDULE_MODE="cosine"
$env:USE_EMA="true"
python train_gpt.py

# Result in: checkpoints_v4/best_ema_model.pt
```

Expected BPB: **1.048-1.055** → **Top 5-15 likely** ✨

---

**More details**: See `QUICK_REFERENCE.md` or `UPGRADE_GUIDE_V4.md`
