# LeakyReLU² + EMA + QAT + PartialRoPE + LNScale + BigramHash(20480) + MLP3.5x

**val_bpb: TBD** (pending 8xH100 runs) | 8×H100 SXM

## Approach

Our own codebase built from scratch, incorporating proven techniques from the leaderboard:

### Architecture (validated in 25 experiments on RTX 4090)
- **11 layers**, 512d, 8 heads, 4 KV heads
- **LeakyReLU(0.5)²** activation (from PR #493/#518, -0.003 bpb vs relu²)
- **BigramHash(20480)** — larger bigram vocab (validated -0.011 bpb)
- **MLP 3.5x** — wider hidden dim (validated -0.005 bpb)
- **Partial RoPE** (16/64 dims) — from PR #287
- **LN Scale** (1/√(layer+1)) — from PR #287

### Training
- Muon optimizer + AdamW for scalars/embeddings
- **EMA(0.997)** weight averaging (replaces SWA, better for quantization)
- **Late QAT** at 15% warmdown threshold (from PR #374)
- Warmdown 3500 iters, grad clip 0.3

### Quantization & Eval
- Mixed int5(MLP)/int6(attn) quantization + zstd-22
- Sliding window eval stride=64

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=20480 MLP_MULT=3.5 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Experiment Evidence

See EXPERIMENT_RESULTS.md for full 25-experiment ablation study.
