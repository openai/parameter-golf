# Quick Start: Overnight Training on Laptop

## One-Liner

```bash
cd /Users/akhileshgogikar/parameter-golf/records/track_10min_16mb/2026-03-25_Ternary_Feedback_TTT && \
source OVERNIGHT_LAPTOP_CONFIG.sh && \
python train_gpt_mlx.py 2>&1 | tee training.log &
```

Then run in separate terminal:
```bash
tail -f training.log
```

---

## Setup (First Time Only)

1. **Ensure data is available:**
   ```bash
   ls -lh ./data/datasets/fineweb10B_sp8192/fineweb_train_*.bin
   ```

2. **If data missing, adjust paths:**
   ```bash
   export DATA_PATH="/path/to/fineweb10B_sp8192"
   export TOKENIZER_PATH="/path/to/fineweb_8192_bpe.model"
   ```

3. **Verify MLX is installed:**
   ```bash
   python -c "import mlx.core as mx; print(mx.__version__)"
   ```

---

## Run Overnight

### Step 1: Start Training (before bed)
```bash
cd /Users/akhileshgogikar/parameter-golf/records/track_10min_16mb/2026-03-25_Ternary_Feedback_TTT
source OVERNIGHT_LAPTOP_CONFIG.sh
nohup python train_gpt_mlx.py > training_$(date +%s).log 2>&1 &
```

### Step 2: Optional - Monitor from Phone
```bash
# Leave this command running in a terminal
watch -n 30 'tail -20 training_*.log'
```

### Step 3: Check Results Next Morning
```bash
# See final metrics
tail -50 training_*.log

# Extract BPB
grep "final_sliding" training_*.log
grep "lawa:\|swa:" training_*.log

# See where model was saved
grep "Artifact saved" training_*.log
```

---

## What to Expect

| Time | Loss | BPB | Notes |
|------|------|-----|-------|
| Start | 3.5-4.0 | - | Initial random weights |
| 1 hour | 3.0-3.2 | - | Warmup phase |
| 3 hours | 2.5-2.7 | ~2.6 | Active learning |
| 6 hours | 2.1-2.3 | ~2.2 | Mid-training |
| 10 hours | 2.0-2.1 | ~2.0 | Converging |
| 12 hours | 1.95-2.05 | **~1.95** | Final (LAWA/SWA applied) |

---

## Key Improvements in This Version

✨ **SmearGate** — Temporal position mixing
✨ **LAWA** — Latest-A-Wins averaging (applied at end)
✨ **SWA** — Stochastic Weight Averaging (applied at end)

These are applied **automatically** when training completes. Watch logs for:
```
lawa:applied k=10 snapshots for serialization
```

---

## Troubleshooting

### Training never starts
```bash
# Check data
ls -lh ./data/datasets/fineweb10B_sp8192/fineweb_train_*.bin
```

### Errors about tokenizer
```bash
# Check tokenizer path
ls -lh ./data/tokenizers/fineweb_8192_bpe.model
```

### GPU OOM (shouldn't happen on MLX, but if training is slow)
```bash
# Reduce batch size
export TRAIN_BATCH_TOKENS=262144
```

### Want to stop training early
```bash
pkill -f "python train_gpt_mlx.py"
# Training will still export model on exit
```

---

## What Gets Created

```
logs/overnight_laptop_<timestamp>/
├── overnight_laptop_mlx_<timestamp>.txt      # Training log
└── overnight_laptop_mlx_<timestamp>_model.ternary.ptz  # Final model (16MB)
```

The `.ptz` file is the compressed ternary model ready for evaluation.

---

## Next Steps After Training

1. **Evaluate the model:**
   ```bash
   export MODEL_PATH="logs/overnight_laptop_*/overnight_laptop_mlx_*_model.ternary.ptz"
   python eval_on_testset.py --model "$MODEL_PATH"
   ```

2. **Compare to baseline:**
   - CUDA version: 1.9538 BPB (best)
   - Expected MLX: ~1.95-2.05 BPB

3. **If results are good:**
   - Back up the config and logs
   - Consider longer overnight run (18-24 hours)
   - Apply same techniques to CUDA version

---

## Pro Tips

**Monitor without terminal:**
```bash
# Run training in background, check anytime
screen -S training
# Then in screen: source OVERNIGHT_LAPTOP_CONFIG.sh && python train_gpt_mlx.py
# Detach with Ctrl-A then D
# Later: screen -r training
```

**Capture metrics to file:**
```bash
# Extract key metrics automatically
watch -n 300 'grep "val_bpb\|final_" training_*.log >> metrics.txt'
```

**Optimize power usage:**
```bash
# Reduce CPU usage (slower but cooler)
# In OVERNIGHT_LAPTOP_CONFIG.sh, try:
# export MAX_WALLCLOCK_SECONDS=86400  # 24 hours instead of 12
# export TRAIN_BATCH_TOKENS=262144   # Halve again
```

---

## Support

If training fails or gets stuck:
1. Check logs: `tail -100 training_*.log | grep -i error`
2. Verify data integrity: `ls -lh ./data/datasets/fineweb10B_sp8192/ | wc -l`
3. Check disk space: `df -h .`
4. Reduce batch size and retry

The code is tested and should complete successfully on M1/M2/M3 MacBooks with 8GB+ memory.
