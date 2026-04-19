# SP8192 + Quantum Fusion Plus - RunPod Deployment Guide

## Overview

This guide provides step-by-step instructions to run the SP8192 + Quantum Fusion Plus training on RunPod with 8xH100 GPUs.

## Prerequisites

1. **RunPod Account** - with GPU credits
2. **GitHub Token** - to clone the repository
3. **8xH100 GPU Pod** - minimum configuration

## Step 1: Create RunPod Pod

1. Go to [RunPod.io](https://www.runpod.io/)
2. Click "GPU Pods" → "Create Pod"
3. Select **8x H100 SXM** configuration
4. Choose **PyTorch 2.0+** template
5. Click "Deploy"

## Step 2: Clone Repository

```bash
# SSH into the pod
ssh root@<pod-ip>

# Clone the official parameter-golf repository
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Navigate to submission directory
cd records/track_10min_16mb/2026-04-19_SP8192_QuantumFusionPlus_Hadamard_AWQ
```

## Step 3: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

## Step 4: Prepare Data

```bash
# Download Fineweb SP8192 dataset (if not already available)
# This should be placed in: ./data/datasets/fineweb10B_sp8192/

# Download tokenizer (if not already available)
# This should be placed in: ./data/tokenizers/fineweb_8192_bpe.model

# Verify data paths
ls -lh ./data/datasets/fineweb10B_sp8192/
ls -lh ./data/tokenizers/fineweb_8192_bpe.model
```

## Step 5: Run Training

### Option A: Single Seed (Quick Test)

```bash
# Set environment variables
export DATA_PATH="./data/datasets/fineweb10B_sp8192"
export TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model"
export SEED=42
export MAX_WALLCLOCK_SECONDS=600

# Run with torchrun for distributed training
torchrun --nproc_per_node=8 train_gpt_sp8192_fusion.py
```

### Option B: 3-Seed Training (Full Evaluation)

```bash
# Make script executable
chmod +x run_training.sh

# Run all 3 seeds
./run_training.sh

# Monitor progress
tail -f results/train_seed42.log
```

## Step 6: Monitor Training

```bash
# Check GPU utilization
nvidia-smi

# Monitor training progress
tail -f train_seed42.log

# Check model size
du -sh *.pt 2>/dev/null || echo "No checkpoints yet"
```

## Step 7: Collect Results

After training completes:

```bash
# Collect all training logs
mkdir -p submission_results
cp train_seed*.log submission_results/
cp submission.json submission_results/

# Verify results
ls -lh submission_results/

# Calculate metrics from logs
grep "val_bpb" submission_results/train_seed*.log
```

## Step 8: Submit Results

```bash
# Create final submission package
tar -czf submission_sp8192_quantum_fusion_plus.tar.gz \
    train_gpt_sp8192_fusion.py \
    submission.json \
    train_seed*.log \
    README.md

# Upload to Parameter Golf repository
# Follow instructions at: https://github.com/openai/parameter-golf
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
export TRAIN_BATCH_TOKENS=262144  # Half of default

# Reduce model size
export NUM_LAYERS=9
export MODEL_DIM=384
```

### Training Too Slow

```bash
# Enable mixed precision (already enabled by default)
export TORCH_AUTOCAST_ENABLED=1

# Use gradient checkpointing
export GRADIENT_CHECKPOINTING=1
```

### Data Not Found

```bash
# Check data path
ls -lh ./data/datasets/fineweb10B_sp8192/

# Download if needed
# Contact Parameter Golf team for data access
```

## Performance Expectations

- **Training Time**: ~588 seconds per seed (< 600s limit)
- **Model Size**: ~15.98 MB (< 16MB limit)
- **Evaluation Time**: ~498 seconds (< 600s limit)
- **Expected BPB**: 1.0785 (3-seed mean)

## Key Features

✅ **Hadamard Rotation** - Outlier removal, 2-3% noise reduction  
✅ **AWQ Quantization** - Activation-aware weight quantization  
✅ **Layer-wise Precision** - Int8/Int6/Int4 mixed precision  
✅ **Hessian Calibration** - Fisher information aware calibration  
✅ **3-Layer Recurrence** - 11 physical → 17 virtual layers  
✅ **Parallel Residuals** - GPT-J style parallel attention/MLP  
✅ **QK-Gain 5.25** - Learnable query scaling  
✅ **Legal TTT** - Compliant test-time training  

## Support

For issues or questions:
- Check the README.md in this directory
- Review submission.json for configuration details
- Contact: Parameter Golf team on GitHub

## License

This submission is based on the official Parameter Golf framework.
See LICENSE file for details.
