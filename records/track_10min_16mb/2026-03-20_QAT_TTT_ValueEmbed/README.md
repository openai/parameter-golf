PR #315 base (1.1248) + Online Logit Bias. Non-record pending compute.

## what's new

**online logit bias (OLB)** (`OLB_LR=0.1`): online learned bias vector added to logits during sliding window eval. Updated via exact CE gradient: `bias -= lr * (softmax(logits+bias) - onehot(target))`. Learns the optimal logit correction online as it processes the validation set. Zero model parameters, near-zero compute, doesn't modify model weights. Strictly better than frequency counting.

## base (matching PR #315 SOTA)

11L, 512d, 3x MLP, int6+zstd, SmearGate, BigramHash(2048), OrthoInit+muP, FA3, seq2048+NTK RoPE, slide s64, Muon WD 0.04, XSA last 4 layers, EMA decay=0.997, Partial RoPE (16/64 dims), LN Scale, Late QAT.

## command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
TRAIN_BATCH_TOKENS=524288 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
OLB_LR=0.1 OLB_MOMENTUM=0.9 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## run plan ($25 = 5 runs)

1. `OLB_LR=0 SEED=1337` -> reproduce PR #315 baseline, expect 1.1248
2. `OLB_LR=0.1 SEED=1337` -> test OLB
3. `OLB_LR=0.05 SEED=1337` -> test lower OLB if 0.1 is too aggressive
4. Best config `SEED=42` -> second seed
5. Best config `SEED=2025` -> third seed

## toggle

`OLB_LR=0` disables OLB.
