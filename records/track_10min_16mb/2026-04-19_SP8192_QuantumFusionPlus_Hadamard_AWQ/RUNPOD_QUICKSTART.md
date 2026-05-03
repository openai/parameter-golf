# RunPod Quick Start Guide

## Complete Automated Deployment in 5 Steps

### Step 1: Create RunPod Pod

1. Go to [RunPod Console](https://console.runpod.io/deploy)
2. Click "GPU Pods" → "Create Pod"
3. **Select Configuration**:
   - GPU: **8x H100 SXM** (or similar)
   - Container: **PyTorch 2.0+** (any recent PyTorch image)
   - Volume: **500GB** (for data + training)
4. Click "Deploy"
5. Wait for pod to start (usually 2-3 minutes)

### Step 2: SSH into Pod

```bash
# Get the SSH command from RunPod console
# It will look like:
ssh -i ~/.ssh/runpod_key root@<pod-ip>

# Or use password if available
ssh root@<pod-ip>
```

### Step 3: Download Submission Files

```bash
# Clone the official repository
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Navigate to submission directory
cd records/track_10min_16mb/2026-04-19_SP8192_QuantumFusionPlus_Hadamard_AWQ

# Make scripts executable
chmod +x *.sh
```

### Step 4: Set GitHub Token (Optional but Recommended)

```bash
# For automatic upload of results to GitHub
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Or you can provide it interactively during deployment
```

### Step 5: Run Complete Deployment

```bash
# One-command execution
bash deploy.sh

# Or run individual steps:
# bash setup_runpod.sh
# bash download_data.sh
# bash train_all_seeds.sh
# bash collect_results.sh
# bash upload_to_github.sh
```

## What Happens During Deployment

### Phase 1: Setup (5 minutes)
- Install Python dependencies
- Verify CUDA/GPU setup
- Create directories
- Clone official repository

### Phase 2: Data Download (10-20 minutes)
- Download Fineweb SP8192 dataset
- Download tokenizer
- Verify data integrity

### Phase 3: Training (60-90 minutes)
- Run seed 42 training (~20 minutes)
- Run seed 314 training (~20 minutes)
- Run seed 999 training (~20 minutes)
- Collect training logs

### Phase 4: Results Collection (5 minutes)
- Extract metrics from logs
- Verify submission files
- Create summary report

### Phase 5: Upload (5 minutes)
- Push results to GitHub
- Update PR #1732
- Verify upload

**Total Time: ~2-2.5 hours**

## Monitoring Progress

### In Another Terminal

```bash
# SSH into pod
ssh root@<pod-ip>

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f /root/logs/train_seed42.log
tail -f /root/logs/train_seed314.log
tail -f /root/logs/train_seed999.log

# Check disk usage
df -h
```

### Expected Output

During training, you should see:
```
[INFO] Starting SP8192 + Quantum Fusion Plus training
[INFO] Device: cuda, Seed: 42
[INFO] Model: 11L x 512d, vocab=8192
[INFO] Hadamard: True, AWQ: True
[INFO] Hessian: True, Layer-wise: True
[INFO] Model created with 3,100,000 parameters
[INFO] Training setup complete. Ready to start training.
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
export TRAIN_BATCH_TOKENS=262144  # Half of default

# Reduce model size
export NUM_LAYERS=9
export MODEL_DIM=384
```

### Training Too Slow

```bash
# Verify GPU usage
nvidia-smi

# Check if all 8 GPUs are being used
# Should show 8 processes if using torchrun correctly
```

### Data Download Fails

```bash
# Try manual download
cd /root/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp8192

# Or download from alternative source
# Contact Parameter Golf team for data access
```

### Upload to GitHub Fails

```bash
# Verify token
echo $GITHUB_TOKEN

# Check git configuration
git config user.email
git config user.name

# Manual push
cd /root/parameter-golf
git push origin submission/sp8192-quantum-fusion-plus
```

## After Deployment

### Check Results

```bash
# View results summary
cat /root/results/results_summary.txt

# View individual logs
cat /root/results/train_seed42.log
cat /root/results/train_seed314.log
cat /root/results/train_seed999.log

# Check submission files
ls -lh /root/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_QuantumFusionPlus_Hadamard_AWQ/
```

### Verify PR Update

```bash
# Check if PR was updated
curl -s https://api.github.com/repos/openai/parameter-golf/pulls/1732 | grep -o '"updated_at":"[^"]*"'

# Or visit: https://github.com/openai/parameter-golf/pull/1732
```

## Performance Expectations

| Metric | Expected | Actual |
|--------|----------|--------|
| val_bpb | 1.0785 | TBD |
| Training Time | ~588s | TBD |
| Evaluation Time | ~498s | TBD |
| Model Size | ~15.98 MB | TBD |

## Tips for Success

1. **Use 8xH100 or better** - Smaller GPU counts will be slower
2. **Allocate 500GB volume** - Data + training artifacts need space
3. **Monitor GPU usage** - Should see 80-90% utilization
4. **Keep terminal open** - Don't close SSH during training
5. **Save logs** - Download logs before stopping pod
6. **Verify results** - Check results_summary.txt before uploading

## Need Help?

- **GitHub Issues**: https://github.com/openai/parameter-golf/issues
- **Parameter Golf Docs**: https://github.com/openai/parameter-golf/blob/main/README.md
- **RunPod Support**: https://www.runpod.io/support

## Cleanup

After deployment completes:

```bash
# Download results locally
scp -r root@<pod-ip>:/root/results /local/path/

# Stop the pod (to save credits)
# Go to RunPod console and click "Stop"

# Delete the pod (if no longer needed)
# Go to RunPod console and click "Delete"
```

---

**Good luck with your training! 🚀**
