# How to Run Parameter Golf Models

**Environments covered**:
- 🔵 **Google Colab** (free GPU, but limited)
- 🍎 **Local Mac M4** (CPU/MLX — slow, but good for testing)
- 🔴 **RunPod 8×H100** (for actual submissions)

---

## Option 1: Google Colab (Free, Limited GPU)

### ⚠️ Important Limitations
- Colab GPUs are **not H100s** (T4 or L4, much slower)
- **Won't reach 1.1147 BPB** (different hardware, shorter time)
- **Good for**: Testing code, debugging, small experiments
- **Not for**: Official submissions (need 8×H100)

### Step 1: Create Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. Create new notebook: **File** → **New notebook**
3. Name it: `parameter-golf-test`

### Step 2: Clone Repository

Paste this in the first cell:

```python
# Clone the repo
!git clone https://github.com/yourusername/parameter-golf.git
!cd parameter-golf && pwd
```

Run it (Shift+Enter). If you haven't pushed to GitHub yet, upload the files manually:

```python
# Upload files manually
from google.colab import files
files.upload()  # Select your parameter-golf folder
```

### Step 3: Install Dependencies

```python
# Install PyTorch (CPU version for Colab is fine for testing)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install sentencepiece zstandard numpy

# Verify
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Step 4: Set Up Data (Minimal)

```python
import os
os.chdir('/content/parameter-golf')

# Create minimal fake data (for testing only, won't give real BPB)
!mkdir -p data/datasets/fineweb10B_sp1024
!mkdir -p data/tokenizers

# Create tiny dummy data files
!head -c 100000 /dev/urandom > data/datasets/fineweb10B_sp1024/fineweb_train_0.bin
!head -c 100000 /dev/urandom > data/datasets/fineweb10B_sp1024/fineweb_val_0.bin

# Copy tokenizer (or download from HF)
!wget -q https://huggingface.co/meta-llama/Llama-2-7b/resolve/main/tokenizer.model -O data/tokenizers/fineweb_1024_bpe.model || \
echo "Note: Tokenizer download may fail. Use CPU/dummy tokenizer for testing."
```

### Step 5: Run Model #001 (Test Version)

```python
import os
os.chdir('/content/parameter-golf')

# Set environment variables for a shorter test run
os.environ['SEED'] = '42'
os.environ['ITERATIONS'] = '100'  # Much shorter than 20000 for testing
os.environ['VAL_LOSS_EVERY'] = '10'
os.environ['NUM_LAYERS'] = '9'    # Smaller model for Colab
os.environ['MODEL_DIM'] = '256'
os.environ['TRAIN_BATCH_TOKENS'] = '65536'  # Much smaller batch
os.environ['MAX_WALLCLOCK_SECONDS'] = '300'  # 5 min max

# Run training (this will be slow on CPU, but shows it works)
!python train_gpt.py
```

### Step 6: View Results

```python
# Find the final loss in the output
# Look for: "Final Sliding BPB: X.XXXX"
# (Will be much worse than 1.1147 on Colab CPU, which is expected)

print("✅ Training complete! Check output above for final BPB.")
print("Note: Colab CPU won't match H100 results. Use for testing only.")
```

---

## Option 2: Local Mac M4 (CPU/MLX)

### ⚠️ Important Limitations
- **Very slow** (M4 CPU will take hours)
- **Good for**: Testing code changes, debugging, small models
- **Not for**: Getting competitive BPB (need GPU)
- **MLX option**: Faster than CPU, but still not competitive

### Step 1: Clone Repository (Locally)

```bash
# Navigate to where you keep code
cd ~/dev  # or wherever

# Clone
git clone https://github.com/yourusername/parameter-golf.git
cd parameter-golf
```

If not pushed to GitHub yet, just navigate to your existing folder.

### Step 2: Install Dependencies

#### Option A: CPU-Only PyTorch

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CPU)
pip install torch torchvision torchaudio
pip install sentencepiece zstandard numpy

# Verify
python3 -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### Option B: MLX (Apple Silicon, Faster)

```bash
# MLX is optimized for Apple Silicon
pip install mlx

# Also install standard deps
pip install sentencepiece zstandard numpy

# Verify
python3 -c "import mlx.core as mx; print(f'MLX available: {mx.default_device()}')"
```

### Step 3: Download Data (Minimal)

The full FineWeb dataset is **large**. For local testing, use tiny data:

```bash
# Create data directories
mkdir -p data/datasets/fineweb10B_sp1024
mkdir -p data/tokenizers

# Option A: Use tiny random data (for testing only)
python3 << 'EOF'
import numpy as np
import os

# Create tiny dummy files
for i in range(2):
    data = np.random.randint(0, 1024, (100, 1024), dtype=np.int32)
    with open(f'data/datasets/fineweb10B_sp1024/fineweb_train_{i}.bin', 'wb') as f:
        f.write(data.tobytes())
    with open(f'data/datasets/fineweb10B_sp1024/fineweb_val_{i}.bin', 'wb') as f:
        f.write(data.tobytes())
print("✅ Created dummy data files")
EOF

# Download tokenizer (SentencePiece, needed for train_gpt.py)
python3 << 'EOF'
import sentencepiece as spm

# Create a minimal tokenizer for testing
spm.SentencePieceTrainer.train(
    input=__file__,  # dummy input
    model_prefix='data/tokenizers/fineweb_1024_bpe',
    vocab_size=1024,
    model_type='bpe'
)
print("✅ Created tokenizer")
EOF
```

### Step 4: Run Model #001 (Test Version)

```bash
# Navigate to repo
cd parameter-golf

# Set smaller parameters for local testing
export SEED=42
export ITERATIONS=50          # Very small for testing
export NUM_LAYERS=6           # Much smaller
export MODEL_DIM=128          # Tiny
export TRAIN_BATCH_TOKENS=1024  # Tiny batch
export MAX_WALLCLOCK_SECONDS=60
export VAL_LOSS_EVERY=5

# Run (will be slow on CPU, ~1-5 min for 50 iterations)
python3 train_gpt.py
```

### Step 5: View Results

```bash
# Check the output for:
# "Train loss: X.XXX" at each step
# "Val loss: X.XXX | Val BPB: X.XXXX" (if val_loss_every triggered)

# The final BPB will be bad (not competitive), but shows it works
echo "✅ Training complete!"
```

---

## Option 3: RunPod 8×H100 (Real Submissions)

### This is Where You Submit

**Only use RunPod for actual submission attempts.**

### Step 1: Set Up RunPod Pod

1. Go to [RunPod](https://runpod.io)
2. Select **GPU Cloud** → **GPU Pod**
3. Choose: **8×A100 80GB** or **8×H100 80GB SXM**
4. Click **Connect** → Opens terminal

### Step 2: Clone Repository

```bash
git clone https://github.com/yourusername/parameter-golf.git
cd parameter-golf
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install FA3 (required for SOTA)
pip install flash-attn --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291

# Other deps
pip install sentencepiece zstandard numpy

# Verify
python3 -c "import torch; from flash_attn_interface import flash_attn_func; print('✅ All deps OK')"
```

### Step 4: Mount Data

```bash
# If you have data on RunPod persistent storage:
ln -s /workspace/data/fineweb10B_sp1024 ./data/datasets/fineweb10B_sp1024
ln -s /workspace/data/tokenizers ./data/tokenizers

# Or download from cloud
aws s3 cp s3://your-bucket/fineweb10B_sp1024 ./data/datasets/fineweb10B_sp1024 --recursive
```

### Step 5: Run Model #001 (Full)

```bash
# Print the launch command
python3 models/model_001_sota_baseline/para.py

# Copy the command for seed 314, paste and run:
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 ... torchrun --standalone --nproc_per_node=8 train_gpt.py

# Wait ~10 minutes for training to complete
```

### Step 6: Record Results

```bash
# When complete, note:
# - Final Sliding BPB (e.g., 1.1147)
# - Artifact size (e.g., 15.86 MB)
# - Steps taken (e.g., 6927)

# Edit docs/training-log.md locally and commit
```

---

## Quick Comparison Table

| Environment | Speed | Cost | GPU Type | Good For |
|-------------|-------|------|----------|----------|
| **Google Colab** | Slow | Free | T4/L4 | Testing code, debugging |
| **Mac M4 CPU** | Very slow | $0 | CPU only | Testing locally, no GPU |
| **Mac M4 MLX** | Slow | $0 | Apple Silicon | Faster local testing |
| **RunPod 8×H100** | Fast ⭐ | $$/hour | H100 | Real submissions (SOTA) |

---

## Troubleshooting

### Google Colab

**Problem**: `No module named 'torch'`
```bash
# Solution: Reinstall in cell
!pip install --upgrade torch torchvision torchaudio
```

**Problem**: Out of memory
```python
# Reduce batch size in environment
os.environ['TRAIN_BATCH_TOKENS'] = '8192'
```

### Mac M4

**Problem**: `ModuleNotFoundError: No module named 'sentencepiece'`
```bash
pip install sentencepiece
```

**Problem**: Very slow training
```bash
# Reduce model size even more
export NUM_LAYERS=3
export MODEL_DIM=64
export TRAIN_BATCH_TOKENS=256
```

**Problem**: Can't find data files
```bash
# Make sure data directory structure is correct
ls -la data/datasets/fineweb10B_sp1024/
# Should show: fineweb_train_*.bin, fineweb_val_*.bin
```

### RunPod

**Problem**: Flash Attention error
```bash
# Reinstall FA3 for your CUDA version
pip install flash-attn --no-build-isolation
```

**Problem**: `torchrun: command not found`
```bash
# Make sure PyTorch is installed
python3 -c "import torch.distributed"
# If fails, reinstall torch
```

---

## Workflow Summary

| Goal | Environment | Steps |
|------|-------------|-------|
| **Test code locally** | Mac M4 CPU | Clone → Install → Run tiny model (50 iterations) |
| **Quick debugging** | Google Colab | Upload → Install → Run test |
| **Real submission** | RunPod 8×H100 | Clone → Install → Run full 20K iterations |
| **Experiment locally** | Mac M4 MLX | Create model_NNN → Edit para.py → Run 1 seed |
| **Official PR** | RunPod 8×H100 | Run 3 seeds → Calculate stats → Create PR |

---

## Next Steps

1. **Choose your environment**:
   - 🍎 Mac M4 → Start with Option 2 (CPU), test your changes
   - ☁️ Colab → Option 1, if you need a GPU
   - 🔴 RunPod → Option 3, only for final submissions

2. **Follow the step-by-step instructions above**

3. **When it works**:
   - Colab: You'll see training loss decrease
   - Mac M4: Same, but slower
   - RunPod: You'll get BPB ≈ 1.1147 (if SOTA config)

4. **Create model_002+ to experiment** (same environment, faster iteration)

5. **When confident**: **Move to RunPod for official submission**

---

**Last updated**: 2026-04-14
