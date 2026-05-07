# 12L INT4 bQAT + EMA Fix + Deterministic QAT

**val_bpb: 1.1594** (seed 3, full TTT) | **15.97 MB** | 8×H100 SXM

## Results (8×H100 80GB SXM)

| Seed | steps | Pre-quant val/bpb | Post-quant bpb | Post-TTT bpb | QAT trigger | Artifact |
|------|-------|-------------------|----------------|--------------|-------------|----------|
| 1    | 5021  | 1.1683            | 1.1703         | ~1.165       | 65% wallclock | 15,899,385 |
| 3    | —     | 1.1729            | 1.2002         | **1.1594**   | 65% wallclock | 15,967,640 |

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 12 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | 10240 buckets, INT4 bQAT |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| Skip | U-Net skip connections |
| resid_mix | Learnable x/x₀ blend (always active) |
| Weight avg | EMA(0.997) with QAT-activation reset |
| Quantization | INT4 MLP + INT4 bigram + INT6 attn + zstd |
| QAT trigger | Wallclock fraction (65% of budget) |
| TTT | Legal score-first, lr=0.002, 3 epochs |

## Key Innovations

### 1. INT4 Bigram QAT (novel)

Standard submissions quantize the bigram hash table to INT6 at export. This submission trains the bigram embedding and projection weights with INT4 STE fake-quantize during QAT, then exports at INT4 (clip=7). This saves ~370KB compressed vs INT6, enabling 12 layers to fit inside 16MB.

No prior competition submission has quantized the bigram table below INT6.

### 2. EMA Reset at QAT Activation

The core quantization-quality bug in naive EMA+LateQAT:

- EMA accumulates weights over all training steps with decay=0.997
- If QAT runs only in the last N steps, the exported EMA weights are still partially pre-QAT
- INT4 with non-QAT-adapted weights → large quantization error

Fix: `_enable_qat()` resets `ema_state = None`, restarting EMA from the clean QAT checkpoint. After N more QAT steps, EMA is 100% QAT-adapted. Result: quantization degradation drops from +0.193 BPB (proxy_v3) to +0.002 BPB (proxy_v4).

### 3. Deterministic Wallclock QAT Trigger

Standard LR-scale QAT trigger fires when `lr_scale < threshold`. On multi-GPU runs, early step timing variance (NCCL sync, torch.compile recompiles) causes the `step_ms` estimate to spike → `warmdown_ms` overestimates → LR scale appears low early → QAT fires prematurely with an undertrained model.

Fix: `LATE_QAT_FRAC=0.65` fires QAT when `elapsed_ms >= 0.65 × max_wallclock_ms`, giving deterministic QAT activation at ~390s regardless of step count variance. Falls back to LR-scale method when no wallclock cap is set (proxy runs, ablations).

## Run Command

```bash
SEED=1 LATE_QAT_FRAC=0.65 VAL_LOSS_EVERY=1000 \
NUM_LAYERS=12 MLP_QUANT_BITS=4 XSA_LAST_N=4 EMA_ENABLED=1 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.9 TTT_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Training Curve (8×H100, seed 1)

| Step | val_bpb (fake-quant) | Notes |
|------|---------------------|-------|
| 1000 | 1.3030 | |
| 2000 | 1.2481 | |
| 3000 | 1.2128 | |
| 3500 | — | QAT activated (step_avg jumps to ~122ms) |
| 4000 | 1.1924 | post-QAT activation |
| 5000 | 1.1683 | wallclock cap hit |
| 5021 | 1.1683 | final step |
| **Post-quant (no TTT)** | **1.1703** | +0.002 degradation only |
| **Post-quant (TTT ~1.165)** | **~1.165** | TTT eval partial (58% complete) |

## Size Budget

| Component | Compressed bytes |
|-----------|----------------|
| Model (int4/int6/zstd) | 15,820,803 |
| Code | 78,582 |
| **Total** | **15,899,385** |

Budget: 16,777,216 bytes (16MB) — **877KB margin**

## Credits

- LeakyReLU² activation: PR #493, PR #518
- XSA (Cross-layer Shared Attention): PR #414
- EMA weight averaging: PR #374
- Legal TTT recipe: PR #461
- INT5/INT6 QAT with STE: PR #317, PR #374
- BigramHash embedding: PR #320
- U-Net skip connections: PR #363
- resid_mix: prior work in this repo
