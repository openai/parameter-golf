# Record: Int6 + MLP 3x + NorMuon + Weight Decay

**val_bpb: 1.1875** | **Total size: 15,669,844 bytes** (under 16MB)

Four improvements over the baseline:

1. **Int6 per-row quantization** ([-31, 31] range, stored as int8) replaces int8 quantization. Per-row fp16 scales with 99.99 percentile clipping. Frees ~25% artifact space vs int8, enabling MLP 3x. Tied embedding kept in fp16 (most quantization-sensitive tensor). Quantization gap: only 0.0042 BPB.

2. **MLP 3x expansion** (hidden=1536 vs baseline 1024). Enabled by int6 space savings. More expressive nonlinear feature transformation between attention operations.

3. **NorMuon optimizer** — per-row normalization before Newton-Schulz orthogonalization ensures each row contributes equally to the gradient update, preventing large-norm rows from dominating.

4. **Decoupled Muon weight decay** (0.02) — regularizes weights toward zero, producing a tighter distribution that quantizes and compresses better.

Additional hyperparameter tuning: MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03, MUON_MOMENTUM=0.99 (warmup from 0.92 over 1500 steps), WARMDOWN_ITERS=3000, GRAD_CLIP_NORM=1.0.

## Configuration

```
NUM_LAYERS=9
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=3
VOCAB_SIZE=1024
TIE_EMBEDDINGS=1
TRAIN_BATCH_TOKENS=524288
TRAIN_SEQ_LEN=1024
MATRIX_LR=0.02
SCALAR_LR=0.02
TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99
MUON_WEIGHT_DECAY=0.02
WARMDOWN_ITERS=3000
GRAD_CLIP_NORM=1.0
```

## Command

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=official_8xh100_9L_int6_mlp3x \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1833 |
| Post-quant roundtrip val_bpb | **1.1875** |
| Quantization gap | 0.0042 BPB |
| Steps | 12,514 |
| Step avg | 47.95ms |
| Total tokens | ~6.56B |
| Artifact size | 15,669,844 bytes |
| Peak memory | 11,319 MiB |

Training progression:
- step 2000: val_bpb 1.3127
- step 4000: val_bpb 1.2666
- step 6000: val_bpb 1.2548
- step 8000: val_bpb 1.2452
- step 10000: val_bpb 1.2348
- step 12000: val_bpb 1.1923
- step 12514: val_bpb 1.1833 (wallclock cap)

Trained and evaluated on 8xH100 SXM (RunPod).

## Included Files

- `train_gpt.py` — Training script with int6 quantization, MLP 3x, NorMuon, weight decay
- `submission.json` — Leaderboard metadata
- `train.log` — Full training log from 8xH100 run
