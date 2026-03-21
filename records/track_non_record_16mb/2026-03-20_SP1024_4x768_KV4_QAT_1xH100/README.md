This is a non-record submission run on 1xH100 SXM 80GB (10-minute wallclock).

## Approach

The key insight is that the baseline's 9-layer 512-dim architecture is suboptimal for the parameter budget. A wider, shallower 4-layer 768-dim model with grouped-query attention (12 heads, 4 KV heads) achieves better loss at equivalent wallclock time by putting more parameters into each forward pass — essentially better L(N) optimization.

### Architecture Changes
- **4 layers x 768 dim** instead of 9 layers x 512 dim
- **12 attention heads, 4 KV heads** (GQA) — optimal ratio from sweep
- **QK gain = 2.5** (baseline 1.5) — sharper attention improves convergence

### Training Changes
- **Muon matrix LR = 0.06** — baseline 0.04 was undertrained; swept 0.04–0.30 across step budgets
- **Gradient clipping = 0.5** — stabilizes training with higher LR
- **Beta2 = 0.99** — better second-moment tracking
- **QAT via STE** — straight-through estimator quantization-aware training, activated after 0.5% warmup. Closes the int8 quantization gap from ~0.03 bpb to 0.0016 bpb.

### Methodology
- 51 experiments on A40 (5–30 min each) to find optimal architecture and hyperparameters
- 4 runs on 1xH100 (10 min each) to validate and sweep batch size
- Architecture advantage confirmed: 4x768 beats 9x512 by 0.032 bpb at matched wallclock

## Results (1xH100 SXM 80GB, 10 min)

| Run | Batch Tokens | Steps | val_bpb | Artifact |
|-----|-------------|-------|---------|----------|
| Run 1 | 65,536 | 6,938 | 1.3474 | 15.69 MB |
| Run 2 | 131,072 | 6,250 | 1.3165 | 15.72 MB |
| **Run 3** | **262,144** | **4,661** | **1.3043** | **15.10 MB** |
| Previous | 524,288 | 2,365 | 1.3128 | 15.09 MB |

262K batch is the sweet spot on 1 GPU — balances gradient quality vs step count.

### Architecture comparison at matched wallclock (15 min, A40)
| Config | val_bpb | Steps |
|--------|---------|-------|
| **4x768 (ours)** | **1.4168** | 839 |
| 9x512 (baseline) | 1.4490 | 618 |
| Delta | **-0.032** | |

### QAT effectiveness
- Quantization gap without QAT: ~0.03 bpb
- Quantization gap with QAT: 0.0016 bpb
- QAT is activated after 0.5% of training (minimal warmup needed)

## 8xH100 Projection
On 8xH100 at 10 min with default batch (524K tokens):
- ~20,000 steps (vs 2,365 on 1xH100)
- Estimated val_bpb: ~1.19 (beats baseline 1.2244 by ~0.035)
- Pending compute grant for official leaderboard run

## Configuration
```bash
RUN_ID=submission_v1 \
NUM_LAYERS=4 \
MODEL_DIM=768 \
NUM_HEADS=12 \
NUM_KV_HEADS=4 \
QK_GAIN_INIT=2.5 \
MATRIX_LR=0.06 \
GRAD_CLIP_NORM=0.5 \
BETA2=0.99 \
TRAIN_BATCH_TOKENS=262144 \
QAT_ENABLED=1 \
QAT_WARMUP_FRAC=0.005 \
SWA_ENABLED=0 \
torchrun --standalone --nproc_per_node=1 train_gpt_qat.py
```

## Included Files
- `train_gpt.py` — modified training script with QAT, architecture changes, and hyperparameter overrides
- `submission.json` — leaderboard metadata
