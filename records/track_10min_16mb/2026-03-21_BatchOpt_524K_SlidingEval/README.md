# Batch-Optimized 524K + Warmdown 4000

**Mean val_bpb: 1.1497** (3 seeds, sliding window stride=64, post int5/int6+zstd quantization roundtrip)

## Key Changes from SOTA (#1 Entry)

Two hyperparameter changes to the #1 leaderboard script (thwu1), no code changes:

1. **TRAIN_BATCH_TOKENS=524288** (down from 786432): Reduces per-step compute, yielding ~7,300 steps instead of ~5,100 on our hardware. More optimizer updates compensate for smaller batch.

2. **WARMDOWN_ITERS=4000** (up from 3000): With more total steps from the smaller batch, the warmdown schedule needed retuning. WD=4000 gives a smoother LR decay curve that better matches the 7,300-step training trajectory.

## Results

| Seed | val_loss | val_bpb | Steps | ms/step | Artifact |
|------|----------|---------|-------|---------|----------|
| 1337 | 1.94123297 | 1.14971055 | 7,361 | 81.51 | 15,927,087 |
| 42 | 1.94043174 | 1.14923602 | 7,248 | 82.98 | 15,768,933 |
| 7 | 1.94198904 | 1.15015834 | 7,269 | 82.55 | 15,786,397 |
| **Mean** | **1.94121792** | **1.14970164** | | | |
| **Std** | | **0.00046** | | | |

All 3 artifacts under 16,000,000 bytes.

## Command

```bash
RUN_ID=submission \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 \
TRAIN_BATCH_TOKENS=524288 \
WARMDOWN_ITERS=4000 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Architecture

Identical to #1 entry (thwu1):
- 10 transformer layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x (hidden=1536), relu^2, SmearGate, BigramHash(10240)
- Orthogonal init, U-Net skips, tied embeddings
- Muon WD=0.04, momentum=0.99, SWA(frac=0.4, every=50)
- Mixed int5(MLP)/int6(attn) + FP16 embed + zstd-22
- Sliding window eval stride=64

## Hardware

8x NVIDIA H100 80GB HBM3 SXM (RunPod Parameter Golf template). PyTorch 2.9.1+cu128.
