# Sliding Window Eval + Long-Context Training

## Score
**val_bpb = 1.1793** (baseline: 1.2244, improvement: 0.045 BPB / 0.081 nats)

Sliding window evaluation with stride=512 on a model trained at seq_len=4096. Every scored token sees 3584 tokens of context with no positional extrapolation — the model trained at this length.

## Approach

### Training: Long Sequences + High Momentum
Train at `TRAIN_SEQ_LEN=4096` with low learning rate (`MATRIX_LR=0.02`), high Muon momentum (`0.99`), and smaller batch (`393216` tokens). The longer training context teaches the model better long-range dependencies. High momentum smooths the optimization landscape.

### Evaluation: Sliding Window (stride=512)
After training, evaluate with overlapping windows of 4096 tokens, sliding by 512 tokens. Only the last 512 positions of each window contribute to the final score. This ensures every token has at least 3584 tokens of preceding context.

The key insight: training at 4096 means the evaluation windows match the training context exactly. No NTK-RoPE extrapolation is needed — the model has seen these position encodings during training. This eliminates the quality degradation that affects extrapolation-based approaches.

Evaluation takes 117 seconds on 8xH100 (within the separate 10-minute eval budget).

### Quantization
Standard int8 post-training quantization with NTK-aware RoPE for eval context extension. The `WARMDOWN_ITERS=3000` schedule with low LR naturally produces smooth weights that quantize well.

## Configuration

```
TRAIN_SEQ_LEN=4096 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
WARMDOWN_ITERS=3000 TRAIN_BATCH_TOKENS=393216
EVAL_SEQ_LEN=4096 EVAL_STRIDE=512
```

## Results

| Eval Method | val_loss | val_bpb |
|------------|----------|---------|
| Standard (non-overlapping, seq=4096) | 2.0131 | 1.1922 |
| **Sliding window (stride=512, seq=4096)** | **1.9912** | **1.1793** |

- **15.88MB** artifact (under 16MB)
- **600s** training wallclock + **117s** eval on 8xH100 SXM
- **~11,000 steps** at ~54ms/step

## Reproduction

```bash
TRAIN_SEQ_LEN=4096 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 TRAIN_BATCH_TOKENS=393216 \
EVAL_SEQ_LEN=4096 EVAL_STRIDE=512 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Hardware: 8xH100 SXM (RunPod, 54ms/step). Faster hardware will yield more steps and a better score.
