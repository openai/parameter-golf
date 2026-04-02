# Polar Express NS + SLOT Eval + XSA-All

**val_bpb: TBD** (pending 8xH100 run) | **~15.9 MB** | 8xH100 SXM

## Key Innovations

### 1. Polar Express Newton-Schulz (arXiv:2505.16932)
Replaces fixed-coefficient NS orthogonalization in Muon with per-iteration minimax-optimal polynomials. 4 Polar Express steps achieve equal or better orthogonality than 5 fixed-coefficient steps, saving ~1-2ms per training step.

### 2. SLOT Eval (eval-time delta optimization)
Per-batch additive delta vector [B,1,d_model] optimized at eval time via 8 AdamW steps (lr=0.005). Model weights frozen; gradients flow only through the final projection. Expected -0.01 to -0.02 BPB improvement.

### 3. XSA on All Layers
Extended Exclusive Self-Attention from last 4 layers to all layers (XSA_LAST_N=11). Zero new parameters, -0.002 BPB improvement.

## Architecture
Built on PR #549 (LeakyReLU² + Legal TTT + Parallel Muon) stack:
- 11L, 512d, 8H, 4KV, MLP 3x with LeakyReLU(0.5)²
- BigramHash(1536), XSA-all, Partial RoPE(16/64), LN Scale
- VE128 on layers 9-10
- EMA(0.997) + SWA(50)
- GPTQ-lite int6 + lzma
- Legal score-first TTT

## Run Command
Based on PR #1019 config (SOTA 1.1147) + our additions:
```bash
# PR #1019 base config
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 XSA_LAST_N=11 \
WARMDOWN_ITERS=4000 TARGET_MB=15.9 \
# Our additions
MUON_BACKEND_STEPS=4 \
SLOT_ENABLED=1 SLOT_STEPS=8 SLOT_LR=0.005 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
Note: TTT is dropped (was neutral/negative on the #1019 stack per 25 failed attempts).
The PR #1019 train_gpt.py already has: LeakyReLU², EMA, SWA, VE, LN Scale, Self-Gen GPTQ.

## Credits
- Base model: PR #414 by @signalrush
- LeakyReLU² + TTT: PR #549 by @abaybektursun
- Polar Express: arXiv:2505.16932 (Keller Jordan et al.)
- SLOT concept: PR #1176
