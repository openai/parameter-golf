# LoRA TTT + Stride-32 on SOTA #1 (10L Int5MLP BigramHash SWA)

## Score: val_bpb = pending (estimated ~1.136)

Builds on current SOTA #1 (1.1428 BPB, `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`) by adding two **eval-only** improvements that cost zero artifact bytes:

### 1. LoRA Test-Time Training (+0.003-0.005 BPB)
Per-document rank-8 LoRA adaptation during evaluation. Adapts Q/V projections and LM head. Score-then-train ordering ensures causality (no data leakage). Adapters reset between documents.

### 2. Sliding Window Stride=32 (+0.002-0.004 BPB)
Reduced from stride=64. Each token scored with 2x more preceding context overlap.

### Base Architecture (identical to SOTA #1)
- 10L, dim=512, 8 heads, 4 KV heads, MLP3x (1536 hidden)
- SmearGate + BigramHash(10240, dim=128)
- SWA (every=50, start=40%), Orthogonal init
- Int5 MLP + Int6 attention + FP16 tied embeddings
- zstd-22 compression, U-Net skip connections
- Muon 0.99, WD=0.04, seq2048, batch=786K
- ~23.4M params, ~15.86MB artifact

### Training Config (overrides via env vars)
```
TRAIN_SEQ_LEN=2048
TRAIN_BATCH_TOKENS=524288
WARMDOWN_ITERS=5000
MUON_MOMENTUM=0.95
MATRIX_LR=0.02
SCALAR_LR=0.02
TIED_EMBED_LR=0.03
GRAD_CLIP_NORM=1.0
EMA_DECAY=0.999
EVAL_STRIDE=32
```

### Command
```bash
RUN_ID=v2_seed42 SEED=42 TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
WARMDOWN_ITERS=5000 MUON_MOMENTUM=0.95 EVAL_STRIDE=32 EMA_DECAY=0.999 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Files
- `train_gpt.py` — 1393 lines, based on SOTA #1 + LoRA TTT + stride=32
- `README.md` — this file
- `submission.json` — metadata
- Train logs pending 8xH100 runs
