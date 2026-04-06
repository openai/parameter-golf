# 11L Depth Recurrence + Discriminative Pre-Quant TTT

**val_bpb: 1.0887** (3-seed mean, std 0.0014) | ~15.9 MB | 8xH100 SXM

## Results (8xH100 80GB SXM)

| Seed | Steps | Pre-TTT bpb | Sliding Window bpb | Artifact |
|------|-------|-------------|-------------------|----------|
| 1337 | 6,158 | 1.1399 | **1.08770** | 15,926,365 |
| 42   | 6,158 | 1.1371 | **1.08825** | 15,924,771 |
| 2025 | 6,158 | 1.1367 | **1.09029** | 15,914,559 |
| **Mean** | | | **1.0887 (std 0.0014)** | |

## Key Innovation 1: Depth Recurrence

After the standard 11-layer forward pass, blocks 4 and 5 are run a second time with shared weights. This gives 13 effective layer passes from 11 physical blocks at zero parameter overhead.

```python
# In GPT forward(), after main block loop:
if args.recur_enabled:
    for lid in args.recur_layers:   # [4, 5]
        x = self.blocks[lid](x)     # second pass, same weights
```

Adds ~3ms per step (97ms vs 94ms without), fitting within the 600s training budget.

## Key Innovation 2: Discriminative Pre-Quant TTT

After EMA weight averaging but before GPTQ quantization, AdamW fine-tuning on validation data with per-block linearly-scaled learning rates. Early blocks get 0.3x base LR (preserve learned features), later blocks get 1.0x (full adaptation).

```python
depth_frac = (i - ttt_freeze_blocks) / max(num_trainable_blocks - 1, 1)
lr_scale = 0.3 + 0.7 * depth_frac   # 0.3x at block 0, 1.0x at block 10
```

Config: `ttt_lr=0.0005`, `ttt_epochs=10`, `ttt_cosine_decay=True`, all blocks trainable. Score-first compliant: each chunk scored under `inference_mode()` before being used for training. Post-dTTT bpb: 1.0992-1.1006 (before quantization).

## Compliance

- Score-first TTT: each chunk evaluated under `torch.inference_mode()` before adaptation
- No eval-time adaptation: model frozen after training + dTTT + GPTQ
- No n-gram, no two-pass, no external data lookup
- No tokenizer/dataset modifications

## Training Architecture

Built on PR #1351 base (Christopher-Lee-McClendon), which extends PR #1019 (abaybektursun):

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3x with LeakyReLU(0.5)^2 |
| Depth Recurrence | Blocks 4,5 (2nd pass, shared weights) |
| QK-Gain | 5.0 (learnable per-head scalar on Q) |
| Optimizer | MuonEq-R (row-norm before NS5) |
| Quantization | GPTQ int6 + lzma |
| Weight avg | EMA(0.997) + SWA(every 50) |
| Pre-Quant TTT | dTTT AdamW, 10 epochs, per-block LR scaling |
| BigramHash | 1536 |
| RoPE | Partial (16/64 dims) |
| VE | dim=128, layers 9-10 |

## Run Command

```
RECUR_ENABLED=1 RECUR_LAYERS=4,5
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10
TTT_ENABLED=1 TTT_LR=0.0005 TTT_EPOCHS=10
TTT_FREEZE_BLOCKS=0 TTT_GRAD_CLIP=1.0 TTT_COSINE_DECAY=1 TTT_BATCH_SEQS=32
MUON_WD=0.04 ADAM_WD=0.04
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=599 EVAL_STRIDE=64
QK_GAIN_INIT=5.0 SEED=1337
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Timing Budget

| Phase | Time |
|-------|------|
| Training | ~599s |
| Standard eval (roundtrip + sliding window) | ~130s |
| Total eval | ~130s (< 10 min) |

## Credits

- Discriminative TTT + MuonEq-R + QK-Gain: PR #1351 by @Christopher-Lee-McClendon
- AR Self-Gen GPTQ base: PR #1019 by @abaybektursun
- Depth recurrence concept: PR #1140, PR #1260, PR #1289
