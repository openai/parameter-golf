# Monitoring Overnight Laptop Training

## What to Expect in the Logs

The training run will produce real-time logs showing:

### Regular Training Logs
```
step:500/15000 loss:2.3456 t:120000ms step:240ms
step:1000/15000 loss:2.1234 t:300000ms step:240ms
```
- **loss**: Training loss on current batch
- **t**: Total elapsed wall-clock time (milliseconds)
- **step**: Per-step time (milliseconds)

### Validation Logs (every 1000 steps)
```
step:1000/15000 val_loss:2.0987 val_bpb:1.9876 train_time:300000ms
```
- **val_loss**: Cross-entropy loss on validation set
- **val_bpb**: Bits per byte on validation set (final metric)
- Track this curve to see convergence

### Weight Averaging Logs
Watch for these lines indicating LAWA/SWA activation:

#### LAWA Collection (every 200 steps)
```
[Note: Logged internally, not printed unless explicitly enabled]
```
After training completes:
```
lawa:applied k=10 snapshots for serialization
```
- Means 10 weight snapshots were averaged together
- This becomes your final model weights

#### SWA Collection (during warmdown)
```
[Note: Logged internally during warmdown phase]
```
After training completes (if warmdown occurred):
```
swa:applied count=45 for serialization
```
- Means 45 weight accumulations from warmdown phase
- Applied if SWA snapshots > LAWA snapshots

#### EMA Application (fallback)
If neither LAWA nor SWA collected enough:
```
EMA shadow weights applied
```
or
```
EMA shadow was never updated (short run) — using trained weights directly
```

### Final Evaluation Logs
```
final_eval val_loss:1.9876 val_bpb:1.9876
final_sliding val_loss:1.9543 val_bpb:1.9543 (stride=128, T=0.95) eval_time:15000ms
```
- **final_eval**: Single-pass evaluation
- **final_sliding**: Sliding window evaluation with optimal temperature

### Serialization Summary
```
artifact:15.84MB code:125000 total:15950000/16000000 (FITS)
Done. Artifact saved to logs/overnight_laptop_2026xxxx_xxxxxx/run_id_model.ternary.ptz
```
- Confirms that model fits in 16MB budget

---

## Log File Location

Logs are saved to:
```
logs/overnight_laptop_<TIMESTAMP>/<RUN_ID>.txt
```

Example:
```
logs/overnight_laptop_20260402_153245/overnight_laptop_mlx_20260402_153245.txt
```

### Monitor in Real-Time
```bash
tail -f logs/overnight_laptop_*/overnight_laptop_mlx_*.txt
```

### Monitor Specific Metrics
```bash
# Loss convergence
grep "val_bpb" logs/overnight_laptop_*/overnight_laptop_mlx_*.txt

# Weight averaging application
grep "lawa:\|swa:\|EMA shadow" logs/overnight_laptop_*/overnight_laptop_mlx_*.txt

# Final metrics
grep "final_" logs/overnight_laptop_*/overnight_laptop_mlx_*.txt
```

---

## Performance Expectations

### Expected Timeline on M3 MacBook
- **Throughput:** ~1.25 steps/sec (varies by model config)
- **12-hour window:** ~15,000 steps
- **Validation evals:** Every 1000 steps = 15 evals over 12h

### Expected Loss Curve
Based on previous runs with similar architecture:
- **Start:** Loss ~3.5-4.0
- **After 1k steps:** Loss ~2.8-3.0
- **After 5k steps:** Loss ~2.2-2.4
- **After 10k steps:** Loss ~2.0-2.2
- **Final (~15k steps):** Loss ~1.95-2.05
- **Sliding window BPB:** ~1.95-2.05

### What Indicates Success
✓ Loss decreases smoothly week-on-week
✓ BPB reaches 1.95-2.05 range by end
✓ LAWA snapshot averaging applied
✓ Model fits in 16MB budget
✓ No OOM (out-of-memory) errors

### What Indicates Issues
✗ Loss plateaus or increases
✗ BPB > 2.5 by end of run
✗ Training crashes with OOM
✗ Per-step time increases significantly
✗ No weight averaging applied

---

## Optimization Hints

### If Training Too Slow (<0.5 steps/sec)
1. Reduce `TRAIN_BATCH_TOKENS` further (try 262144)
2. Reduce `TRAIN_SEQ_LEN` (try 512)
3. Reduce `NUM_LAYERS` (try 8)

### If Training Too Fast (>3 steps/sec, likely wrong config)
1. Increase `TRAIN_BATCH_TOKENS` (try 524288)
2. Check that CUDA is not being used (should be MLX only)

### If Loss Doesn't Converge Well
1. Increase `WARMUP_STEPS` (try 20)
2. Decrease learning rates (try 0.03, 0.02)
3. Reduce `WARMDOWN_FRACTION` (try 0.5)

### If Loss Oscillates (unstable)
1. Reduce `MATRIX_LR` (try 0.025)
2. Increase `GRAD_CLIP_NORM` (try 0.5)
3. Check power stability (some Macs throttle on battery)

---

## Overnight Safety Checklist

Before starting:
- [ ] Plug in laptop to power (training uses significant CPU)
- [ ] Disable sleep: `System Preferences → Energy Saver → Sleep: Never`
- [ ] Monitor temperature: `top -o %CPU` should show ~70-90% utilization
- [ ] Check disk space: Need ~20GB free for logs, models, temp
- [ ] Note the start time and expected completion time (usually 10-14 hours)
- [ ] Have a backup config saved in case of terminal loss

### Temperature Monitoring (M-series Mac)
Normal: 70-80°C (fan audible but not loud)
Warn: 85-90°C (consider reducing batch size)
Danger: >95°C (training throttles, reduce batch size)

Check with:
```bash
# In another terminal while training
watch -n 5 'sysctl -n machdep.cpu.core_temperature'
```

---

## Comparing to CUDA Version

The CUDA version (train_gpt.py) is trained on H100:
- Achieves ~0.65 BPB gap to community SOTA (1.1147 BPB)
- Trains much faster but same ternary quantization approach

The MLX version optimizes for laptop efficiency:
- 10-12 layer model (vs 12 in CUDA)
- Smaller batch (393k tokens vs 786k)
- Shorter seq (1024 vs 2048)
- Shorter wall-clock (12h vs unlimited)
- **Same TKC architecture and weight averaging strategies**

Expected BPB gap: ~0.1-0.2 BPB worse than CUDA (due to less training, smaller model, but same architecture).

---

## Troubleshooting

### OOM (Out of Memory)
```
RuntimeError: [ALLOCATOR] out of memory
```
**Fix:** Reduce `TRAIN_BATCH_TOKENS` (try 196608), reduce `MODEL_DIM` (try 384), reduce `NUM_LAYERS` (try 8)

### Training Hangs
**Fix:** Check for infinite loops or deadlock in gradient accumulation. Try `MLX_EAGER_EVAL=1` for debugging.

### Slow Training
**Common causes:**
1. Disk thrashing (check `iostat`)
2. Power throttling (plug in + disable sleep)
3. Other processes hogging CPU (close Chrome, Slack, etc.)

### Loss NaN
**Fix:** Check for gradient explosion. Reduce learning rates (try 0.02, 0.015).

---

## Post-Training Analysis

After overnight run completes, analyze:

```bash
# Extract final metrics
grep "final_" logs/overnight_laptop_*/overnight_laptop_mlx_*.txt

# Compare to baseline
# (previous local run metrics go here)

# Verify weight averaging was applied
grep "lawa:\|swa:" logs/overnight_laptop_*/overnight_laptop_mlx_*.txt
```

If BPB is significantly better than baseline:
- Weight averaging improvements confirmed
- Consider using same approach for CUDA version

If BPB is worse:
- Might need hyperparameter tuning
- Check if convergence is just slow (needs more steps)
- Verify TKC features enabled (feedback, capsules, koopman)
