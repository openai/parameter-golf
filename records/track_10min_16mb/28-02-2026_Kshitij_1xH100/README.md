# Parameter Golf Submission — Kshitij

## Overview
This submission explores **high-throughput small-model training** under the 10-minute constraint using a **1x H100 SXM GPU**.

The focus is on:
- Faster convergence via aggressive learning rates
- Reduced model size for better compression
- Efficient token throughput

---

## Key Idea

Instead of scaling model size, this run optimizes:
- **Tokens/sec throughput**
- **Fast convergence dynamics**
- **Smaller architecture → better compression ratio**

---

## Hyperparameters (IMPORTANT)

All hyperparameters are **NOT hardcoded**.

They are passed via **environment variables**, which are visible in logs.

Example (from run):

```bash
# ===== MAX THROUGHPUT =====
export TRAIN_SEQ_LEN=256
export TRAIN_BATCH_TOKENS=131072

# ===== MODEL =====
export NUM_LAYERS=10
export MODEL_DIM=256
export NUM_HEADS=8
export NUM_KV_HEADS=2

# ===== LR =====
export MATRIX_LR=0.035
export SCALAR_LR=0.035

# ===== OTHER =====
export MLP_MULT=4
export QK_GAIN_INIT=1.2

python train_gpt.py

These values are automatically picked up inside:

os.environ.get(...)

This ensures:

Clean separation of config vs code
Easy reproducibility
Transparent tuning
Model Configuration
Parameter	Value
Layers	10
Hidden Dim	256
Heads	8
KV Heads	2
MLP Multiplier	4
Sequence Length	256
Batch Tokens	131072

Total params: ~9.7M

Training Strategy
Short sequence length (256) → faster steps
High LR (0.035) → rapid early convergence
Muon optimizer for matrix updates
Cosine decay (late phase)
Flash attention enabled
Results
Metric	Value
Final val_bpb	1.4618
Training time	~600 seconds
Steps reached	3329
Compression
Artifact	Size
final_model.int8.ptz	10.6 MB
Code	~48 KB
Total	~10.63 MB ✅ (within 16MB limit)
Important Notes
This run uses 1x H100, not 8x (non-record submission)
Configuration is scalable to multi-GPU
Logs clearly show:
hyperparameters
training dynamics
final metrics
Logs

See:

log1.txt (primary run)

Logs include:

environment-based hyperparameters
training progression
validation checkpoints
final compression stats
Code
Based on train_gpt.py baseline
Modified attention + LR schedule

Source:


Final Thoughts

This submission demonstrates that:

Smaller models + high throughput can be competitive
Environment-driven configs improve clarity and reproducibility
Efficient training matters more than brute scale under constraints

---
