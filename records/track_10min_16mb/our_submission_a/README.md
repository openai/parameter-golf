# LeakyReLU² + TTT + BigramHash(20480) + MLP3.5x

**val_bpb: TBD** (pending 8xH100 runs) | 8×H100 SXM

## Changes from PR #549 (LeakyReLU_LegalTTT_ParallelMuon)

Two modifications validated across 5 rounds of experiments on RTX 4090:

### 1. BigramHash(20480) — up from 2048
Larger bigram vocabulary captures more token-pair statistics. Validated at -0.011 bpb improvement over BigramHash(10240) in isolation, with gains additive to other changes.

### 2. MLP 3.5x — up from 3.0x
Wider MLP hidden dimension. Validated at -0.005 bpb in isolation, additive with other improvements.

## Base Stack (from PR #549)
- 11L (512d, 8H, 4KV) with LeakyReLU(0.5)² activation
- XSA on last 4 layers + Partial RoPE (16/64)
- LN Scale (1/√(layer+1)) + VE128 on layers 9-10
- EMA(0.997) + Tight SWA(every 50)
- GPTQ-lite int6 + lzma compression
- Parameter Banking + Parallel Muon optimizer
- Legal score-first TTT (3 epochs SGD, all blocks unfrozen)

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=20480 MLP_MULT=3.5 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Experiment Evidence

See EXPERIMENT_RESULTS.md for full 25-experiment ablation study conducted on RTX 4090.
