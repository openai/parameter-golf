# GPTQLite: Pure Velocity & TTT Preservation

**Target val_bpb: < 1.1194** (beat leaderboard #1)
**Base: 2026-03-23_LeakyReLU_LegalTTT_ParallelMuon → 1.1194 BPB**

## Results (8×H100 80GB SXM)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | TTT time | Artifact |
|------|----------|-------|-------------|-----------------|----------|----------|----------|
| 1337 | 83.87ms  | 7155  | 1.12164     | **1.11901**      | -0.00263 | 421.9s   | 15.851MB |
| 42   | 83.86ms  | 7156  | 1.12229     | **1.11961**      | -0.00268 | 423.2s   | 15.858MB |
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

## Beyond the 10min/16MB Constraint

The features disabled for this submission are **legitimate improvements** that are bottlenecked by the competition's strict budget, not by their quality. In an unconstrained setting (longer training, larger artifact):

| Feature | Why it was disabled | Why it would help with more budget |
|---------|--------------------|------------------------------------|
| **GatedAttention** (PR #841) | +1.5ms/step → costs 130+ training steps in 600s | Per-head sigmoid gates improve attention expressivity; pays off with 30min+ training |
| **ValueResidual** (PR #841) | Same compute overhead | Layer-0 value injection improves gradient flow across deep layers |
| **BigramHash=2048** | Marginal size cost at 16MB | More bigram vocabulary = better subword context modeling |
| **QAT from step 1** (`QAT_ENABLED=1`) | Overhead throughout all ~7000 steps | With more steps, full-run quantization adaptation significantly reduces post-quant degradation |
| **BANK_QAT_THRESHOLD > 0** | Corrupts TTT weights mid-evaluation | With a larger artifact budget and no TTT, Bank QAT enables aggressive int4/int6 compression of a much bigger model |

The 10min/16MB constraint specifically punishes compute overhead and rewards lean, fast models. These features represent the right architectural direction for longer, less constrained training runs.

## Originally Planned Changes vs 2026-03-23 Base (Not Used in Final Submission)

> **Note:** These changes were designed and implemented but ultimately disabled for the final submission. The 10min/16MB constraint made the compute overhead a net negative in this specific setting. They remain in the codebase and are expected to help significantly in longer runs or with larger artifact budgets — see section below.

| Planned Change | Why | Expected Δ BPB | Used? |
|----------------|-----|----------------|-------|
| **GatedAttention=True** | Per-head sigmoid gate (PR #841, nearly no-op init: weight=0, bias=4.0) | -0.002 to -0.005 | ❌ Disabled — +1.5ms/step costs 130+ training steps |
| **ValueResidual=True** | Mixes layer-0 value v0 into all subsequent layers (PR #841, init: λ=[0.5,0.5]) | included above | ❌ Disabled — same overhead |
| **QAT_ENABLED=True from step 1** | int6 fake-quant throughout all ~7000 steps vs only final 15% of warmdown | -0.001 to -0.003 | ❌ Disabled — too much overhead during training |
| **LATE_QAT_THRESHOLD=0.05** | CastedLinear QAT activates in final 5% of warmdown | included above | ❌ Reverted to 0.15 |
| **BigramHash 1536 → 2048** | More bigram vocabulary = better subword context modeling | -0.001 to -0.002 | ❌ Kept at 1536 — marginal size cost at 16MB limit |
| **Total expected gain** | | **-0.004 to -0.010 → ~1.109–1.115 BPB** | — |

## Beyond the 10min/16MB Constraint

The features disabled for this submission are **legitimate improvements** bottlenecked by the competition's strict budget, not by their quality. In an unconstrained setting (longer training, larger artifact):

| Feature | Why it was disabled | Why it would help with more budget |
|---------|--------------------|------------------------------------|
| **GatedAttention** (PR #841) | +1.5ms/step → costs 130+ training steps in 600s | Per-head sigmoid gates improve attention expressivity; pays off with 30min+ training |
| **ValueResidual** (PR #841) | Same compute overhead | Layer-0 value injection improves gradient flow across deep layers |
| **BigramHash=2048** | Marginal size cost at 16MB | More bigram vocabulary = better subword context modeling |
| **QAT from step 1** (`QAT_ENABLED=1`) | Overhead throughout all ~7000 steps | With more steps, full-run quantization adaptation significantly reduces post-quant degradation |
| **BANK_QAT_THRESHOLD > 0** | Corrupts TTT weights mid-evaluation | With a larger artifact budget and no TTT, Bank QAT enables aggressive compression of a much bigger model |

The 10min/16MB constraint specifically punishes compute overhead and rewards lean, fast models. These features represent the right architectural direction for longer, less constrained training runs.

### Headroom & Scaling Evidence

The final submission sits at **~15.851MB** — leaving ~149KB of the 16MB budget unused. Attempts to fill that headroom by increasing `BIGRAM_VOCAB_SIZE` to 1664 and then 2048 produced worse BPB scores and pushed the artifact over the 16MB limit, confirming the model is already well-optimized for this constraint.

This unused headroom, combined with the disabled features (GatedAttention, ValueResidual, BigramHash=2048, BANK_QAT_THRESHOLD), represents a clear scaling path: in an uncapped scenario (larger artifact budget + longer training), all of these levers can be opened simultaneously, and the model would be expected to perform significantly better than the current 1.119x BPB.

## Credits

- **LeakyReLU² activation**: PR #493 by @parinzee, PR #518 by @sofiabod
- **Optimizer (Parameter Banking + Parallel Muon)**: PR #399 by @abaybektursun
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
- **Base model**: PR #414 by @signalrush
