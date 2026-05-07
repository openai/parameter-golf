# SDPA-Compatible 11L + 1-sqrt Cooldown — Blackwell & H100

**val_bpb: 1.1445** (sliding window, stride 64) | **15.68 MB** | 8xH100 SXM, 600s | No TTT

## Two Novel Contributions

### 1. SDPA Fallback for Non-Hopper GPUs
The current record entries all require FlashAttention 3 (Hopper sm_90 only). Our `flash_attn_3_func` wrapper auto-detects FA3 availability and falls back to `F.scaled_dot_product_attention` with proper tensor layout transposition (FA3: B,T,H,D -> SDPA: B,H,T,D) and GQA head expansion. Enables training on Blackwell (sm_121a), AMD, and any GPU with PyTorch SDPA support.

### 2. 1-sqrt Warmdown Schedule
Replaced linear warmdown with `1 - sqrt(t)` where `t` is progress through the warmdown phase. This schedule decays faster initially (more aggressive cooling) but holds higher LR longer before dropping to zero. On single GB10 at 7000 steps: **-0.019 BPB improvement** over linear warmdown (1.4170 vs 1.4362 int6 roundtrip).

## Results

### 8xH100 SXM (786K batch, seq_len 2048, SDPA — no FA3)
| Seed | Steps | ms/step | Pre-quant BPB | Int6 BPB | **Sliding BPB** | Artifact |
|------|-------|---------|---------------|----------|-----------------|----------|
| 1337 | 4,026 | 149 | 1.1637 | 1.1679 | **1.1445** | 15,684,720 |

Note: SDPA is ~70% slower than FA3 on H100 (149ms vs ~87ms/step), limiting us to 4,026 steps vs ~6,900 with FA3. With FA3, the 1-sqrt schedule would likely achieve **~1.10-1.11 BPB**.

### Single GB10 Blackwell (16K batch, seq_len 1024)
| Seed | Steps | ms/step | Pre-quant BPB | Int6 BPB | Sliding BPB | Artifact |
|------|-------|---------|---------------|----------|-------------|----------|
| 1337 | 7,000 | 646 | 1.4166 | 1.4173 | 1.3810 | 13,646,113 |

### 1-sqrt Cooldown Ablation (GB10, 7000 steps)
| Schedule | Int6 BPB | Delta |
|----------|----------|-------|
| Linear warmdown (baseline) | 1.4362 | — |
| **1-sqrt warmdown** | **1.4170** | **-0.019** |

### Hyperparameter Ablation (GB10, 500-step screening)
| Experiment | val_bpb @500 | Notes |
|-----------|-------------|-------|
| Baseline (QK 1.5, WD 0.04) | ~1.85 | Default record hyperparams |
| QK-Gain 3.0 | 1.8164 | Best single change at 500 steps |
| QK-Gain 2.0 | 1.8184 | Close to 3.0 |
| QK-Gain 5.0 | 1.8308 | Top entries use 5.0-5.25 |
| WD 0.09 | 1.8149 | Marginal at 500, worse at 7000 |
| QK 3.0 + WD 0.09 (7000 steps) | 1.4800 | Worse — WD too aggressive for small batch |
| 1-sqrt + batch warmup | 1.8185 | No improvement over 1-sqrt alone |
| 1-sqrt + PTQ 5% | 1.4179 | No improvement |
| 1-sqrt + PTQ + batch warmup | 1.4165 | Marginal, within noise |

### Negative Results: Novel Architectures
- **SSM-Attention Hybrid** (5 conv + 6 attn): -0.24 BPB worse. Conv layers less expressive at 512-dim.
- **Mixture of Experts** (Top-1, 2 experts): -0.52 BPB worse. Incompatible with Muon optimizer + GPTQ pipeline.
- **Self-Distillation** (prev-step KL loss): No improvement. Destabilized training loss.

## Architecture (unchanged from record)
- 11 layers, 512 dim, 8 heads / 4 KV heads (GQA)
- 3x MLP with LeakyReLU(0.5)^2
- BigramHash(3072x112), SmearGate, XSA-all, U-Net skips
- Partial RoPE (16/64 dims), LN Scale, VE128 (layers 9-10)
- Full Hessian GPTQ int6 (AR self-gen calibration), LZMA-9
- EMA(0.997) + Tight SWA(every 50)
- Parallel Muon optimizer

## Run Command
```bash
# 8xH100 (with SDPA fallback — works without FA3)
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=3500 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Single GPU development (any GPU with PyTorch SDPA)
TORCHDYNAMO_DISABLE=1 ITERATIONS=7000 TRAIN_BATCH_TOKENS=16384 \
TRAIN_SEQ_LEN=1024 EVAL_SEQ_LEN=1024 BIGRAM_VOCAB_SIZE=3072 \
BIGRAM_DIM=112 WARMDOWN_ITERS=3500 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Lineage
```
PR #549 (1.1194 BPB) -> Record entry (1.1147 BPB)
    └── This work:
        ├── SDPA fallback for non-Hopper GPUs (Blackwell sm_121a, H100 without FA3)
        ├── 1-sqrt warmdown schedule (-0.019 BPB on GB10)
        ├── H100 validation: 1.1445 BPB sliding (4,026 steps with SDPA)
        ├── Extensive hyperparameter ablation (QK-Gain, WD, batch warmup, PTQ)
        └── Negative results (SSM hybrid, MoE, self-distillation)
```
