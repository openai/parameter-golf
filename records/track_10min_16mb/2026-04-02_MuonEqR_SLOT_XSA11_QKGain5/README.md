# MuonEq-R + Context-Only SLOT + XSA-all + QK-Gain 5.0

**val_bpb: TBD** (3-seed mean) | **~16.0 MB** | 8xH100 SXM

## Changes from PR #549 (1.1194 BPB)

| Change | Expected Impact | Source |
|--------|----------------|--------|
| **MuonEq-R** | -0.001 BPB | arXiv:2603.28254, PR #1260 |
| **Context-Only SLOT** | -0.006 BPB | PR #1217 |
| **XSA all 11 layers** | -0.001 BPB | PR #1019 |
| **QK_GAIN_INIT=5.0** | -0.001 BPB | PR #1217 sweep |
| **Total expected** | **-0.009 BPB** | **~1.110 BPB** |

## MuonEq-R

Row-normalizes gradient matrices before Newton-Schulz orthogonalization (arXiv:2603.28254). Equalizes row norms so the NS iteration operates on a better-conditioned matrix. Zero-byte cost, ~0.001 BPB improvement.

## Context-Only SLOT (Causal)

Per-batch additive delta vector (512 dims) optimized with AdamW during eval. For each sliding window (seq_len=2048, stride=64):

1. Hidden states computed under `torch.no_grad()` — model weights frozen
2. Delta optimized using cross-entropy on **context positions only** (0 to seq_len-stride). The 64 new tokens being scored are excluded from the loss.
3. Final logits computed with optimized delta. NLL recorded for the 64 new positions.

Delta is re-initialized to zeros for each window. Gradient flows only through the linear projection + softcap — not the transformer.

| Parameter | Value |
|-----------|-------|
| Delta shape | (1, 1, 512) |
| Optimizer | AdamW |
| Learning rate | 0.005 |
| Steps | 8 |

## Architecture

PR #549 stack with Parallel Muon:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3x with LeakyReLU(0.5)^2 |
| BigramHash | 1536 |
| XSA | **All 11 layers** (was last 4) |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| QK Gain | **5.0** (was 1.5) |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | **MuonEq-R** + Parallel Muon |

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=11 \
QK_GAIN_INIT=5.0 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
SLOT_ENABLED=1 SLOT_STEPS=8 SLOT_LR=0.005 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

TBD — pending 3-seed validation on 8xH100.

## Legality

- MuonEq-R: standard optimizer improvement, no rule restriction
- Context-Only SLOT: causal by construction — delta optimized on past tokens only, new tokens excluded from loss
- XSA-all: no new parameters, architectural choice
- QK_GAIN=5.0: hyperparameter choice
- No n-gram cache, no two-pass rescoring, no eval-time GPTQ
- Score-first TTT follows PR #461 legal protocol

## Credits

- **Base model + TTT**: PR #549 (@abaybektursun), PR #414 (@signalrush), PR #461 (@Christopher-Lee-McClendon)
- **MuonEq-R**: arXiv:2603.28254, validated in PR #1260
- **SLOT**: Hu et al. arXiv:2505.12392v2, Context-Only variant from PR #1217 (@dexhunter)
- **QK-Gain sweep**: PR #1217
- **XSA-all**: PR #1019
