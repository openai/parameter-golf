# 🤖 AUTOMATED TRAINING SCRIPTS

**Status**: Ready to run on 8xH100 machine
**Scripts**: 3 options (bash, batch, Python)
**Duration**: ~30-40 minutes total (20-30 min data + 10 min training)

---

## 📋 Prerequisites

Before running any script, ensure your machine has:

✅ **Hardware**
- 8x H100 SXM GPUs (80GB each)
- 500GB+ free disk space (for data)
- 64GB+ RAM

✅ **Software**
- Ubuntu 20.04+ OR Windows 10+
- NVIDIA CUDA 12.1+
- Python 3.8+
- git

✅ **Verification**
```bash
nvidia-smi                    # Should show 8 H100s
python --version              # Should be 3.8+
git --version                 # Any recent version
```

---

## 🚀 Option 1: Python Script (Recommended)

**Most portable, works on Linux and Windows**

```bash
# In directory with train_gpt.py:
python train_automated.py
```

### What it does:
1. ✅ Checks GPU availability (8xH100)
2. ✅ Installs dependencies (torch, sentencepiece, numpy)
3. ✅ Clones official repository
4. ✅ Downloads FineWeb dataset (~20-30 min)
5. ✅ Verifies training code
6. ✅ Runs training on 8 GPUs (~10 min)
7. ✅ Verifies model artifact
8. ✅ Saves logs

### Output:
```
✓ 8 GPUs ready
✓ All dependencies installed
✓ Data ready: 8 train files, 1 val file
✓ Training code verified
✓ Training completed in 598.5 seconds
✓ Model artifact: 14.2 MB (limit: 16 MB)
✓ Log file: logs/training_20260322_153045.log
```

---

## 🐧 Option 2: Bash Script (Linux/Mac)

**For Unix-like systems**

```bash
# Make executable
chmod +x train_automated.sh

# Run
bash train_automated.sh
```

### Features:
- ✅ Colored output for clarity
- ✅ Step-by-step progress
- ✅ Comprehensive error checking
- ✅ Automatic logging

### Example run:
```
[STEP 1/5] Verifying environment...
✓ Found 8 GPUs

[STEP 2/5] Installing dependencies...
✓ Dependencies installed

[STEP 3/5] Preparing FineWeb dataset...
  Downloading FineWeb data (this takes 20-30 minutes)...
✓ Data ready: 8 train files, 1 val files

[STEP 4/5] Setting up training code...
✓ Training code verified

[STEP 5/5] Starting training (max 600 seconds)...
```

---

## 🪟 Option 3: Windows Batch Script

**For Windows 10/11 with GPU**

```batch
REM Run from Command Prompt or PowerShell
train_automated.bat
```

### Features:
- ✅ Windows-compatible paths
- ✅ Step-by-step verification
- ✅ Timestamped logs
- ✅ Error handling

### To run from PowerShell:
```powershell
.\train_automated.bat
```

---

## 📊 Detailed Execution Timeline

### Python/Bash/Batch (all identical flow):

```
START
│
├─ [1 min] Verify 8 GPUs present
├─ [1 min] Install dependencies
├─ [20-30 min] Download FineWeb data (8GB, one-time)
├─ [1 min] Setup training code
├─ [10 min] Run training on 8xH100
│         • GPU utilization: 100%
│         • Training losses decrease
│         • Model checkpoints saved
├─ [1 min] Verify model artifact
└─ [DONE] Logs saved, ready for submission

TOTAL: ~33-42 minutes (depending on data download speed)
```

---

## 📂 What Gets Created

After running any script:

```
parameter-golf/
├── data/
│   ├── datasets/fineweb10B_sp1024/
│   │   ├── fineweb_train_0001.bin
│   │   ├── fineweb_train_0002.bin
│   │   └── ... (8 total)
│   └── tokenizers/fineweb_1024_bpe.model
│
├── final_model.int8.ptz            ← Your trained model (~14 MB)
├── logs/
│   └── training_20260322_153045.log ← Training output
│
└── train_gpt.py
```

---

## 🔍 Understanding the Logs

After training, check the log file:

```bash
# View final results
tail -20 logs/training_*.log

# Look for BPB score
grep -i "bpb" logs/training_*.log

# Check training time
grep -i "wallclock\|time" logs/training_*.log

# View all metrics
cat logs/training_*.log | grep -E "loss|bpb|val|step"
```

---

## ✅ Success Checklist After Running

```
After script completes:
☐ final_model.int8.ptz exists
☐ File size < 16 MB
☐ logs/training_*.log exists
☐ Log file shows completion
☐ BPB score is reasonable (≤4.2)
☐ GPU memory returned to normal
☐ No error messages
```

---

## 🐛 Troubleshooting

### Problem: "CUDA not found"
```bash
# Check GPU drivers
nvidia-smi

# Install CUDA if needed
# Ubuntu: sudo apt-get install nvidia-utils
# Windows: Download from https://developer.nvidia.com/cuda-downloads
```

### Problem: "Out of memory"
```bash
# Reduce batch size (edit train_gpt.py or set env var)
TRAIN_BATCH_TOKENS=262144 python train_automated.py
```

### Problem: "Data download slow"
```bash
# Pre-download data separately
python data/cached_challenge_fineweb.py --variant sp1024
# Then run script again
```

### Problem: Script hangs
```bash
# Check GPU status in another terminal
nvidia-smi -l 1

# If stuck, terminate
# Linux/Mac: Ctrl+C then pkill -f torchrun
# Windows: Ctrl+C then taskkill /F /IM python.exe
```

### Problem: "All-reduce failed"
```bash
# Check NCCL debugging
NCCL_DEBUG=INFO python train_automated.py

# Or try simpler version:
NCCL_IB_DISABLE=1 python train_automated.py
```

---

## 📝 After Training: Next Steps

Once training completes successfully:

### 1. Record Results
```bash
# View final metrics
tail -50 logs/training_*.log

# Note the BPB score and time
```

### 2. Update submission.json
```bash
# Edit submission.json with actual metrics
vi submission.json
# Update: val_bpb, training_time_seconds, model_size_mb
```

### 3. Create Results File
```bash
cat > RESULTS.md << EOF
# Training Results

- BPB: [your_score]
- Time: [seconds]s
- Size: [MB]MB
- Log: logs/training_*.log

EOF
```

### 4. Commit to GitHub
```bash
git add final_model.int8.ptz submission.json RESULTS.md
git commit -m "Add trained model and results"
git push
```

### 5. Submit PR
See GITHUB_SETUP.md for complete submission workflow

---

## 🎯 Script Comparison

| Feature | Python | Bash | Batch |
|---------|--------|------|-------|
| **Platform** | Linux/Mac/Windows | Linux/Mac | Windows |
| **Setup** | None | chmod +x | None |
| **Error handling** | Excellent | Good | Good |
| **Portability** | Best | Good | Windows only |
| **Dependencies** | Python | bash | cmd.exe |
| **Recommended** | ✅ | ✅ | ✅ |

---

## 🔧 Customization

### Override defaults (for Python script):
```python
# Edit train_gpt.py or set environment variables:
export ITERATIONS=15000           # Fewer iterations
export WARMDOWN_ITERS=800         # Faster warmdown
export TRAIN_BATCH_TOKENS=262144  # Smaller batches
python train_automated.py
```

### Override defaults (for Bash script):
```bash
# Edit train_gpt.py first, then run
ITERATIONS=15000 bash train_automated.sh
```

### Override defaults (for Batch script):
```batch
REM Edit environment in train_automated.bat or set before:
set ITERATIONS=15000
train_automated.bat
```

---

## 📊 Performance Expectations

If script runs successfully, you should see:

```
GPU Utilization:    95-100% all 8 GPUs
Memory per GPU:     65-75 GB
Training Loss:      Steadily decreasing
Validation BPB:     3.8-4.2 range
Total Time:         550-600 seconds
Final Model:        14-15 MB
```

---

## ✨ Features of These Scripts

✅ **Automated** - No manual intervention needed
✅ **Robust** - Comprehensive error checking
✅ **Logged** - Full output saved for debugging
✅ **Verified** - Results validated before completion
✅ **Clear** - Progress shown in real-time
✅ **Fast** - Optimized for 8xH100
✅ **Complete** - End-to-end from data to model

---

## 🎯 Which Script to Use?

| Your Setup | Best Choice |
|-----------|------------|
| Linux/Mac server | Bash script |
| Windows with GPU | Batch script |
| Cloud VM (any OS) | Python script |
| Unsure | Python script |

---

## 📞 Support

If script fails:

1. Check TESTING.md troubleshooting section
2. Review logs: `cat logs/training_*.log`
3. Check CODE_REVIEW.md for code issues
4. Run with verbose output: `NCCL_DEBUG=INFO python train_automated.py`

---

**Ready to run? Pick your script and start training! 🚀**

```bash
# Option 1 (Recommended - works everywhere)
python train_automated.py

# Option 2 (Linux/Mac)
bash train_automated.sh

# Option 3 (Windows)
train_automated.bat
```

Training will complete in ~30-40 minutes and save results automatically!
