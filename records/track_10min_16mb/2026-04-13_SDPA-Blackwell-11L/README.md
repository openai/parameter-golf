# SDPA-Compatible 11L Record Fork — Blackwell GPU Training

## Approach

Adapted the current record entry ([PR #549](https://github.com/openai/parameter-golf/pull/549) lineage, 1.1147 BPB on 8xH100) to run on non-Hopper GPUs by replacing FlashAttention 3 with PyTorch's native `scaled_dot_product_attention` (SDPA). This enables training and validation on NVIDIA Blackwell (sm_121a), AMD, and any GPU with PyTorch SDPA support.

Validated on a single NVIDIA GB10 (DGX Spark, 128GB unified memory) with reduced batch size.

### Key Contribution: SDPA Fallback for Non-Hopper GPUs

The current record entries all require FlashAttention 3 (Hopper sm_90 only). Our `flash_attn_3_func` wrapper auto-detects FA3 availability and falls back to `F.scaled_dot_product_attention` with proper tensor layout transposition (FA3: B,T,H,D -> SDPA: B,H,T,D) and GQA head expansion.

### Architecture (unchanged from record)
- 11 layers, 512 dim, 8 heads / 4 KV heads (GQA)
- 3x MLP with LeakyReLU(0.5)^2
- BigramHash(3072x112), SmearGate, XSA-all, U-Net skips
- Partial RoPE (16/64 dims), LN Scale, VE128 (layers 9-10)
- Full Hessian GPTQ int6 (AR self-gen calibration), LZMA-9
- EMA(0.997) + Tight SWA(every 50)
- Parallel Muon optimizer

## Results

### Single GB10 (16K batch tokens, seq_len 1024)
| Seed | Steps | ms/step | Pre-quant BPB | **Int6 BPB** | Artifact |
|------|-------|---------|---------------|-------------|----------|
| 1337 | 7,000 | 639 | 1.4357 | **1.4362** | 13,847,230 |

### Hyperparameter Ablation (500-step runs on GB10)
| Experiment | val_bpb @500 | Notes |
|-----------|-------------|-------|
| Baseline (QK 1.5, WD 0.04) | ~1.85 | Default record hyperparams |
| QK-Gain 5.0 | 1.8308 | Top entries use 5.0-5.25 |
| QK-Gain 3.0 | 1.8164 | Best single change at 500 steps |
| QK-Gain 2.0 | 1.8184 | Close to 3.0 |
| WD 0.09 | 1.8149 | Marginal at 500, worse at 7000 |
| QK 3.0 + WD 0.09 (7000 steps) | 1.4800 | Worse than baseline — WD too aggressive for small batch |
| SSM-Attention Hybrid | 2.0867 | Conv layers less expressive |
| Mixture of Experts (2) | 2.3738 | Incompatible with bank optimizer |
| Self-Distillation | 1.8165 | No improvement |

### Negative Results: Novel Architectures
- **SSM-Attention Hybrid** (5 conv + 6 attn layers): GatedConvBlock replaces attention in early layers. 12% faster per step but -0.24 BPB worse. Conv layers lack the expressiveness of attention at 512-dim scale, and unused attention bank parameters waste the artifact budget.
- **Mixture of Experts** (Top-1, 2 experts): Expert weights bypass the Muon optimizer (use Adam instead) and GPTQ Cholesky fails on 3D expert weight tensors. The bank-based architecture is tightly coupled to its optimizer and quantization pipeline.
- **Self-Distillation** (previous-step KL loss): Destabilizes training loss (4->8+) without improving val_bpb. Previous-step logits from noisy small-batch training are not useful soft targets.
- **Hyperparameter tuning at scale**: QK-Gain and WD improvements at 500 steps did not hold at 7000 steps — the original record hyperparams (QK 1.5, WD 0.04) are already optimal for this architecture. Lesson: always validate at full training length.

## Hardware Notes
- **NVIDIA GB10 (sm_121a)**: CUDA capability 12.1 — not officially supported by PyTorch (max 12.0), but works with warnings. FlashAttention 3 unavailable.
- **SDPA Performance**: ~640ms/step vs ~87ms/step on 8xH100 with FA3 (~7.4x slower, consistent with single GPU vs 8 GPU scaling).
- **Memory**: 1,388 MiB peak allocated (model fits easily in 128GB unified memory).
- **Thermal**: GB10 ran at sustained load for 75+ minutes without thermal throttling.

## Expected 8xH100 Performance
Based on the record entry achieving 1.1147 BPB with identical architecture and 786K batch tokens, this SDPA-compatible version should achieve **~1.11-1.12 BPB** on 8xH100 (the SDPA fallback is only active when FA3 is unavailable — on Hopper, FA3 is used directly).

## Run Command
```bash
# Single GPU (development/validation — any GPU with PyTorch SDPA)
TORCHDYNAMO_DISABLE=1 ITERATIONS=7000 TRAIN_BATCH_TOKENS=16384 \
TRAIN_SEQ_LEN=1024 EVAL_SEQ_LEN=1024 BIGRAM_VOCAB_SIZE=3072 \
BIGRAM_DIM=112 WARMDOWN_ITERS=3500 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 (submission — uses FA3 automatically if available)
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage
```
PR #549 (Legal SOTA, 1.1194) — Parallel Muon base
    └── Record entry (1.1147) — AR self-gen GPTQ + XSA-all + BigramHash 3072x112
        └── This work adds:
            ├── SDPA fallback for non-Hopper GPUs (Blackwell sm_121a verified)
            ├── GQA head expansion in SDPA path
            ├── Hyperparameter ablation (QK-Gain sweep, WD tuning)
            └── Negative results (SSM hybrid, MoE, self-distillation)
```
