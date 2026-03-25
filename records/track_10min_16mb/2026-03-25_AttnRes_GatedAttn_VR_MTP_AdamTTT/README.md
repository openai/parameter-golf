# GatedAttn + ValueResidual + MTP + lzma-9

**val_bpb: pending** | **~16 MB** | 8×H100 SXM

Built directly on the 2026-03-23 SOTA (1.1194). No AttnRes — clean base.

## Changes vs 2026-03-23

| Change | Before → After | Artifact cost | Notes |
|--------|---------------|---------------|-------|
| `MTP_NUM_HEADS=1` | 0 → **1** | **0 bytes** (stripped at export) | Stronger training gradient |
| `GATED_ATTENTION=1` | 0 → **1** | ~24KB | Per-head sigmoid gate, bias init=4.0 |
| `VALUE_RESIDUAL=1` | 0 → **1** | ~0 bytes (22 params) | vr_lambda init [0.0, 1.0] |
| `lzma preset=9` | 6 → **9** | 0 bytes (slower compress only) | Free artifact savings |
| `TTT_USE_ADAM` | — | 0 bytes | Added but **off by default** (SGD preserved) |

### MTP (Multi-Token Prediction)

`mtp_num_heads=1` adds a next+1 token prediction head during training. Stripped before serialization (`if "mtp_heads" not in k`), so zero artifact cost. Provides a stronger gradient signal to all layers across the 7000-step training run. `mtp_loss_weight=0.2`.

### Gated Attention

Per-head sigmoid gate on attention output: `nn.Linear(512, 8, bias=True)` per block. Bias initialized to 4.0 so sigmoid(4.0)≈0.98 — gates start near-open and training dynamics are undisturbed at init. The model gradually learns to suppress uninformative heads per-token. ~45K extra params, ~24KB in artifact.

### Value Residual

Mixes the first layer's raw V embeddings into every subsequent attention layer:
`v = vr_lambda[0]*v0 + vr_lambda[1]*v`

**Init: `[0.0, 1.0]`** — starts identical to base (no mixing), learns to use v0 gradually. (The default `[0.5, 0.5]` in the original code was reset to neutral to avoid disrupting early training.)

### lzma preset=9

Replaces `preset=6`. Saves 5-25% artifact size at the cost of slower post-training compression (irrelevant to eval timing). Frees budget for the gated attention params.

### TTT_USE_ADAM (off by default)

`TTT_USE_ADAM=0` — SGD+momentum(0.9) is preserved as in the SOTA. Adam TTT can be tested as a follow-up ablation via `TTT_USE_ADAM=1 TTT_ADAM_LR=0.0002`.

## Inherited Stack (2026-03-23)

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + **lzma-9** |
| Optimizer | Parameter Banking + Parallel Muon |
| TTT | Legal score-first, 3ep, SGD, all blocks |

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
GATED_ATTENTION=1 VALUE_RESIDUAL=1 MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.2 \
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

## Ablation targets

- `GATED_ATTENTION=0` — isolate gated attention contribution
- `MTP_NUM_HEADS=0` — isolate MTP training signal
- `VALUE_RESIDUAL=0` — isolate value residual
- `TTT_USE_ADAM=1 TTT_ADAM_LR=0.0002` — test Adam TTT as follow-up

## Credits

- **LeakyReLU²**: PR #493, PR #518
- **Parallel Muon**: PR #399
- **TTT recipe**: PR #461
- **Base model**: PR #414 (signalrush)
