# GPTQLite: Pure Velocity & TTT Preservation

**Target val_bpb: < 1.1194** (beat leaderboard #1)
**Base: 2026-03-23_LeakyReLU_LegalTTT_ParallelMuon → 1.1194 BPB**

## Results (8×H100 80GB SXM)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | TTT time | Artifact |
|------|----------|-------|-------------|-----------------|----------|----------|----------|
| 1337 | 83.87ms  | 7155  | 1.12164     | **1.11901**      | -0.00263 | 421.9s   | 15.851MB |
| 42   | —        | —     | —           | **—**            | —        | —        | —        |
| 2025 | —        | —     | —           | **—**            | —        | —        | —        |
| **Mean** | —   | —     | —           | **—**            | —        | —        | —        |

## Strategy: Pure Velocity & TTT Preservation

The previous strategy was "Maximize Capacity" — stuffing the 16MB limit with extra features (GatedAttention, ValueResidual, BigramHash=2048). Ablations showed those features slow the GPU enough to cost 130+ critical training steps and destabilize TTT.

The new strategy strips the model to its leanest form, eliminating hidden compute and memory bottlenecks so the model gets more training steps in 600s and completes TTT cleanly.

## Key Changes vs Previous Attempts

### 1. Architecture Speed Diet
| Flag | Value | Why |
|------|-------|-----|
| `GATED_ATTENTION` | **0** (disabled) | Adds ~1.5ms/step overhead — costs 130+ training updates over 600s |
| `VALUE_RESIDUAL` | **0** (disabled) | Same overhead, no net gain when combined with the other changes |
| `BIGRAM_VOCAB_SIZE` | **1536** | Keeps artifact lean; 2048 was marginal at best |

### 2. SWA Removed
| Flag | Value | Why |
|------|-------|-----|
| `SWA_ENABLED` | **0** | Every 50 steps it was copying hundreds of MB of tensors GPU→CPU pointlessly — the script uses EMA weights at the end, not SWA. Disabling buys ~30 extra training steps. |

### 3. QAT Simplified
| Flag | Value | Why |
|------|-------|-----|
| `QAT_ENABLED` | not set (off) | Full QAT from step 1 adds math overhead throughout training |
| `LATE_QAT_THRESHOLD` | **0.15** | Quantization activates only in the final 15% of warmdown |
| `BANK_QAT_THRESHOLD` | **0** | Bank QAT was acting as a TTT assassin — snapping finely-tuned FP32 weights back to Int6 mid-evaluation, causing catastrophic forgetting |

## Unchanged

| Feature | Setting |
|---------|---------|
| **Architecture** | 11L, 512d, 8H, 4KV, 3× MLP |
| **Activation** | LeakyReLU(0.5)² — hardcoded |
| **XSA** | Last 4 layers |
| **VE** | dim=128, layers 9,10 |
| **Partial RoPE** | 16/64 dims, NTK scaling |
| **LN Scale** | 1/√(layer+1) |
| **EMA** | decay=0.997, applied at end of training |
| **Quantization** | GPTQ-lite int6 + zstd-22 |
| **Optimizer** | Parallel Muon + Parameter Banking — all LRs/WDs identical |
| **Legal TTT** | score-first, 3 epochs, freeze=0, lr=0.002, SGD+momentum(0.9) |
| **Training** | TRAIN_SEQ_LEN=2048, EVAL_STRIDE=64, WARMDOWN_ITERS=3500 |

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 GATED_ATTENTION=0 VALUE_RESIDUAL=0 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 ROPE_DIMS=16 LN_SCALE=1 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
LATE_QAT_THRESHOLD=0.15 BANK_QAT_THRESHOLD=0 SWA_ENABLED=0 \
TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 \
ITERATIONS=9000 WARMDOWN_ITERS=3500 MAX_WALLCLOCK_SECONDS=600 \
DATA_PATH=$DATA TOKENIZER_PATH=$TOK SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed1337.log
```

## Timing Budget (actual, seed 1337)

| Phase | Time |
|-------|------|
| Training (wallclock cap) | 600s |
| EMA apply + diagnostic eval | ~2s |
| int6 roundtrip eval | ~6s |
| Sliding window eval (2048, stride=64) | ~75s |
| Legal TTT (3ep, all blocks, 2048 ctx) | ~425s |
| **Total eval** | **~508s ✓** |

## Credits

- **LeakyReLU² activation**: PR #493 by @parinzee, PR #518 by @sofiabod
- **Optimizer (Parameter Banking + Parallel Muon)**: PR #399 by @abaybektursun
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
- **Base model**: PR #414 by @signalrush
