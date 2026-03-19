# Long-Context Sliding Window with Optimized Training

## Score
**val_bpb = 1.1764** (sliding window, stride=512)

Baseline: 1.2244. Improvement: 0.048 BPB / 0.087 nats.

## Approach

Train at longer sequences (2048 tokens) with high Muon momentum (0.99), low learning rate (0.02), and tight gradient clipping (0.3). Evaluate with overlapping sliding windows where every scored token sees 1536+ tokens of preceding context.

### Training
- `TRAIN_SEQ_LEN=2048` — longer context during training improves representation quality
- `MATRIX_LR=0.02` with `MUON_MOMENTUM=0.99` — low LR + high momentum for smooth optimization
- `TRAIN_BATCH_TOKENS=786432` — optimal batch size (swept 393K to 1M)
- `GRAD_CLIP_NORM=0.3` — tighter clipping stabilizes long-sequence gradients (swept 0.1 to 1.0)
- `WARMDOWN_ITERS=3000` — standard warmdown (aggressive warmdown hurts with low LR)

### Evaluation
- Sliding window with `EVAL_STRIDE=512` at `EVAL_SEQ_LEN=2048`
- Every scored token sees at least 1536 tokens of context
- No positional extrapolation — evaluation matches training sequence length
- Eval time: ~80 seconds on 8xH100

### Novel Finding: Train Length Doesn't Matter (With Sliding Window)
Training at 2048 vs 4096 gives identical BPB when evaluated with sliding window (1.1764 vs 1.1765). The sliding window already provides long context at eval — the model just needs to learn local patterns. Training at 2048 is strictly preferable because it gets more steps per 10 minutes.

### Novel Finding: Gradient Clipping Sweet Spot
Long-sequence training benefits from a narrow clipping window. Full sweep:

| Clip | BPB |
|------|-----|
| 0.0 | 1.1780 |
| 0.1 | 1.1766 |
| 0.2 | 1.1765 |
| **0.3** | **1.1764** |
| 0.5 | 1.1769 |

## Configuration

```
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 MATRIX_LR=0.02 SCALAR_LR=0.02
TIED_EMBED_LR=0.03 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3
EVAL_SEQ_LEN=2048 EVAL_STRIDE=512
```

## Reproduction

```bash
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 MATRIX_LR=0.02 SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3 \
EVAL_SEQ_LEN=2048 EVAL_STRIDE=512 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**15.88MB** artifact. 8xH100 SXM (RunPod).
