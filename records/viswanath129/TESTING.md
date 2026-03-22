# Testing and Validation Guide

## Pre-Training Checklist

### 1. Environment Verification

```bash
# Check Python version (must be 3.8+)
python --version

# Verify CUDA availability
nvidia-smi

# Check GPU configuration
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Expected output: 8x H100 SXM (80GB each)
```

### 2. Dependency Installation

```bash
# Install required packages
pip install torch sentencepiece numpy

# Verify installations
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import sentencepiece; print('SentencePiece OK')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
```

### 3. Data Preparation

```bash
# Clone official challenge repository
git clone https://github.com/openai/parameter-golf
cd parameter-golf

# Download FineWeb cached dataset (required)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Verify data structure
ls -lh data/datasets/fineweb10B_sp1024/
# Should contain: fineweb_train_*.bin and fineweb_val_*.bin files
```

### 4. Code Syntax Validation

```bash
# Check Python syntax without running
python -m py_compile train_gpt.py

# Check for common issues
python -m pylint --errors-only train_gpt.py  # optional
```

---

## Training Run Checklist

### Pre-Training (Do NOT skip)

- [ ] All 8 H100 GPUs visible via `nvidia-smi`
- [ ] Data files present in `./data/datasets/fineweb10B_sp1024/`
- [ ] Tokenizer present at `./data/tokenizers/fineweb_1024_bpe.model`
- [ ] Sufficient disk space for model artifact (~50MB working space)
- [ ] System has not been shut down recently (CUDA drivers warm)
- [ ] No other GPU processes running
- [ ] Environment variables unset (to use defaults):
  ```bash
  unset DATA_PATH TOKENIZER_PATH RUN_ID SEED
  unset VAL_BATCH_SIZE VAL_LOSS_EVERY TRAIN_LOG_EVERY
  unset ITERATIONS WARMDOWN_ITERS WARMUP_STEPS
  unset TRAIN_BATCH_TOKENS TRAIN_SEQ_LEN MAX_WALLCLOCK_SECONDS
  unset VOCAB_SIZE NUM_LAYERS NUM_KV_HEADS MODEL_DIM NUM_HEADS
  unset MLP_MULT TIE_EMBEDDINGS ROPE_BASE LOGIT_SOFTCAP
  unset USE_SMEARGATE USE_BIGRAMHASH BIGRAM_HASH_SIZE
  unset USE_SENT_LITE SENT_LITE_ALPHA
  ```

### Run Training

```bash
# Method 1: Using run.sh (recommended)
bash run.sh

# Method 2: Direct torchrun
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Method 3: Custom configuration
MAX_WALLCLOCK_SECONDS=600 ITERATIONS=20000 USE_SENT_LITE=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Monitor Training

```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi

# Expected behavior:
# - All 8 GPUs should show ~100% utilization
# - Memory usage: ~60-70GB per GPU
# - Temperature: 50-80°C
```

### Expected Outputs

After successful completion, you should see:

```
✅ Size check PASSED
Model artifact: 12345678 bytes
Code size: 567890 bytes
Total: 12913568 bytes (limit: 16,000,000)
```

---

## Validation Tests

### 1. Model Load and Forward Pass Test

```python
import torch
from train_gpt import GPT, dequantize_state_dict_int8

# Load model
with open("final_model.int8.ptz", "rb") as f:
    quant_obj = torch.load(f)

state = dequantize_state_dict_int8(quant_obj)

# Create model and load weights
model = GPT(
    vocab_size=1024,
    num_layers=9,
    model_dim=512,
    num_heads=8,
    num_kv_heads=4,
    mlp_mult=3,
    tie_embeddings=True,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.5
)
model.load_state_dict(state)

# Test forward pass
x = torch.randint(0, 1024, (1, 1024))
y = model(x, x)
print(f"✅ Forward pass OK: output shape {y.shape}")
```

### 2. Int8 Quantization Round-Trip Test

```python
from train_gpt import quantize_state_dict_int8, dequantize_state_dict_int8

# Existing test in code (lines 1108-1118)
# Automatically run POST-training to verify correctness
```

### 3. BPB Computation Validation

```python
import torch
import math
from train_gpt import build_sentencepiece_luts

# The code includes validation via SentencePiece LUTs
# Tested during eval_val() calls
```

### 4. Distributed Data Loading Test

```bash
# Run with 2 GPUs locally to test DDP logic
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  ITERATIONS=10 MAX_WALLCLOCK_SECONDS=120
```

---

## Troubleshooting

### Issue: "RuntimeError: CUDA out of memory"

**Solution**: Reduce batch size
```bash
TRAIN_BATCH_TOKENS=262144 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Issue: "FileNotFoundError: No files found for pattern"

**Solution**: Verify data download
```bash
# In parameter-golf repo root:
python3 data/cached_challenge_fineweb.py --variant sp1024
ls -lh data/datasets/fineweb10B_sp1024/fineweb_train*.bin
```

### Issue: "ValueError: Unexpected shard header"

**Solution**: Data format mismatch
```bash
# Check first shard header with Python
python3 -c "
import numpy as np
header = np.fromfile('data/datasets/fineweb10B_sp1024/fineweb_train_0001.bin', dtype='<i4', count=3)
print(f'Magic: {hex(header[0])}, Version: {header[1]}, Tokens: {header[2]}')
"
# Expected: Magic: 0x13450520, Version: 1
```

### Issue: "All-reduce operation failed"

**Solution**: DDP connectivity issue
```bash
# Check NCCL debugging
NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Issue: "Training time exceeded 600 seconds"

**Solution**: Optimization needed
- Reduce `ITERATIONS` to ~15000
- Use fewer validation steps: `VAL_LOSS_EVERY=2000`
- Skip warmup period: `WARMUP_STEPS=0` (note: less safe)

---

## Performance Benchmarking

### Baseline Metrics (Expected on 8xH100)

| Metric | Expected | Tolerance |
|--------|----------|-----------|
| Training time | 550-600s | ±20s |
| Model artifact | 13-15 MB | <16 MB |
| Validation BPB | 3.8-4.2 | ±0.1 |
| Tokens/sec/GPU | 2000-3000 | ±500 |
| GPU memory | 65-70 GB | per GPU |

### Profiling

```bash
# Enable PyTorch profiler (advanced)
PYTORCH_ENABLE_DEBUG_TRACE=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## Reproducibility Verification

Run the same training twice with different seed:

```bash
# First run (seed 1337)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Second run (seed 42)
SEED=42 RUN_ID=run_42 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Compare results (should be identical architecture, different initialization)
```

---

## Sign-off Checklist

Before submitting, verify:

- [ ] Training completes in <600s wallclock
- [ ] Artifact file is <16MB
- [ ] Forward pass works on loaded model
- [ ] Int8 quantization round-trip is lossless
- [ ] BPB score is reasonable (within ~0.1 of prior runs)
- [ ] All 8 GPUs were utilized during training
- [ ] Code syntax is valid (py_compile passes)
- [ ] Data files were downloaded correctly
