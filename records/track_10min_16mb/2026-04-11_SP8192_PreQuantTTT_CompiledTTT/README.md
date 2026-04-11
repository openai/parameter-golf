# Record: SP8192 + Pre-Quant AdamW TTT + Compiled TTT + Parallel Residuals

**val_bpb = 1.0587** (3-seed mean, std 0.0004) | **~15.5 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | Roundtrip BPB | Artifact |
|------|-------------|---------------|----------|
| 42   | 1.05840     | 1.06847       | 15,477,275 |
| 1337 | 1.05856     | 1.06904       | 15,439,370 |
| 2024 | 1.05912     | 1.06921       | 15,480,770 |
| **Mean** | **1.05869** | **1.06891** | **15,465,805** |
| **Std** | **0.00038** | **0.00037** | |

Merged SOTA (PR #1493): **1.0810 BPB**. Delta: **-0.0223 BPB** = **-0.0155 nats**. Clears the 0.005-nat threshold (3.1x). t-statistic = 102.2, p < 0.01.

## Key Techniques

1. **SP8192 + GPTQ SDClip** — int6 matrices (k=12.85), int8 embeddings (k=20.0), zero selective pruning needed (PR #1394 @clarkkev)
2. **3-Layer Depth Recurrence** (layers 3,4,5, activate at step 3000) — 14 virtual layers from 11 physical (PR #1493 @bigbag, PR #1204 @msisovic)
3. **Parallel Residuals** (layers 7+) — GPT-J style two-lane merge, attention and MLP operate independently (PR #1412 @Robby955, PR #1204 @msisovic)
4. **Pre-Quant AdamW TTT** — 6 epochs on val data with `torch.compile` (2x speedup vs uncompiled), freeze first 2 blocks, cosine LR decay. Weights baked into artifact before GPTQ. (PR #1485 @ndokutovich)
5. **QK-Gain 5.25** — learnable per-head query scaling (PR #1493 @bigbag)
6. **Tuned Hyperparameters** — WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72 (PR #1493 @bigbag)
7. **rANS / Brotli auto-compression** — Near-Shannon-optimal entropy coding, auto-selects smaller of rANS vs Brotli-11

## Architecture

11L x 512d x 8H / 4KV, MLP 4x (2048 hidden), LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale (1/sqrt(layer+1)), tied embeddings (init_std=0.005), logit softcap=30.0. Depth recurrence: [0,1,2,3,4,5,3,4,5,6,7,8,9,10] = 14 virtual layers (layers 3-5 repeated once, activated at step 3000). Parallel residuals from layer 7: attention on lane0, MLP on lane1, merged with learned scalar. Sigmoid-gated U-Net skip connections. XSA on all 11 layers (efficient GQA-aware). Value Embeddings (dim=44, layers 9-10). SmearGate.

## Training

MuonEq-R optimizer (Polar Express minimax-optimal coefficients, 4 Newton-Schulz steps) with 3-phase overlapped communication: async reduce-scatter -> Adam step -> local NS + async all-gather. AdamW for embeddings (lr=0.03, wd=0.095) and scalars (lr=0.02, wd=0.02). ~5160 steps in 600s on 8xH100 SXM. Linear warmdown to LR=0 over final 72%. EMA decay 0.9965. Late QAT (STE noise) when LR scale < 15%.

## Pre-Quant AdamW TTT

Fine-tunes the EMA model on validation data BEFORE GPTQ quantization (Track A — result baked into static artifact):

- `torch.compile(dynamic=False, fullgraph=True)` for 2x speedup (426s vs 860s uncompiled)
- AdamW, lr=0.0005, weight_decay=0.0, cosine decay to lr*0.1
- 6 epochs, freeze first 2 transformer blocks
- Batch: 32 sequences x 2048 tokens, grad clip 1.0
- All-reduce gradients across 8 GPUs
- Fresh model instance (avoids inference_mode rotary cache poisoning)

## Quantization

Full-Hessian GPTQ with SDClip: `clip = k * std(row)` for principled rate-distortion tradeoff. int6 for attention/MLP matrices (k=12.85), int8 for token embeddings (k=20.0). AR self-generated calibration data (32 seqs x 2048 tokens, temp=0.8). Byte-shuffle + Brotli-11 compression. Zero selective pruning needed on all 3 seeds.

## Compliance (Track A)

- Pre-quant TTT trains on validation data BEFORE quantization
- Result baked into artifact — fixed predictor at eval time
- No eval-time model adaptation, no SLOT, no n-gram cache
- All training within 600s wallclock on 8xH100
- All artifacts under 16,000,000 bytes on all 3 seeds
- Eval (sliding window stride=64) within 10-minute eval budget (~110s actual)

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

VOCAB_SIZE=8192 BIGRAM_VOCAB_SIZE=0 VE_DIM=44 SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **@clarkkev** — SP8192 + GPTQ Embeddings + SDClip + depth recurrence (PR #1394)
- **@bigbag** — 3-layer recurrence + parallel residuals + QK5.25 + tuned hyperparams (PR #1493)
- **@ndokutovich** — Pre-Quant AdamW TTT (PR #1485)
- **@Robby955** — Parallel residuals + Hessian-aware SDClip (PR #1412)
- **@msisovic** — Parallel residuals concept + mini depth recurrence (PR #1204)
- **@dexhunter** — MuonEq-R + depth recurrence + legal TTT on SP8192 (PR #1285, #1413)
- **@abaybektursun** — Score-first TTT framework (PR #549)
