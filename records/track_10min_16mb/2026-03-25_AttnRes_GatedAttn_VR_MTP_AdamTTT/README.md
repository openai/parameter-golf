# AttnRes + Gated Attention + Value Residual + MTP + Adam TTT + lzma-9

**val_bpb: pending** | **~16 MB** | 8×H100 SXM

Built on the 2026-03-24 AttnRes submission. Adds five improvements on top of the AttnRes stack.

## Changes vs 2026-03-24 base

| Change | Default | Cost | Expected gain |
|--------|---------|------|---------------|
| `MTP_NUM_HEADS=1` | 0 → **1** | 0 bytes (stripped at export) | −0.001 to −0.002 bpb |
| `VALUE_RESIDUAL=1` | 0 → **1** | ~22 params (~0 bytes) | −0.0005 to −0.001 bpb |
| `GATED_ATTENTION=1` | 0 → **1** | ~45K params (~24KB) | −0.001 to −0.002 bpb |
| TTT: SGD → Adam | SGD → **Adam** | 0 bytes | better per-chunk adaptation |
| `lzma preset=6 → 9` | 6 → **9** | 0 bytes (slower compress only) | free artifact savings |

### MTP (Multi-Token Prediction)

`mtp_num_heads=1` adds a vocabulary prediction head for token `t+1` during training, producing a stronger gradient signal for all layers. The head is stripped before serialization at line 1804 (`if "mtp_heads" not in k`), so it contributes **zero bytes** to the 16MB artifact. `mtp_loss_weight=0.2` blends the auxiliary loss with the main cross-entropy.

### Gated Attention

`gated_attention=True` adds `nn.Linear(512, 8, bias=True)` per block — a per-head sigmoid gate on attention output. Zero-initialized weights, bias initialized to 4.0 (sigmoid(4.0) ≈ 0.98, so gates start near-open). The model learns to suppress uninformative heads per token. 11 layers × (512×8 + 8) ≈ 45K params; at int6+lzma: ~24KB within budget.

### Value Residual

`value_residual=True` adds `vr_lambda` (2 float32 params) per attention layer. Mixes the first layer's raw V embeddings into every subsequent attention layer via a learned convex combination `lam[0]*v0 + lam[1]*v`. Initialized to `[0.5, 0.5]`. This is a value-stream skip connection complementary to AttnRes on the residual stream.

### Adam TTT

Replaces `SGD(lr=0.002, momentum=0.9)` with `Adam(lr=2e-4, betas=(0.9, 0.999))` for the test-time training phase. Adam's per-parameter adaptive rates handle the varying distribution of each 32K-token validation chunk better than a global SGD step size. The cosine LR schedule across chunks is preserved, now using `ttt_adam_lr` as the base. `TTT_USE_ADAM=0` reverts to SGD for ablation.

### lzma preset=9

`lzma.compress(quant_raw, preset=9)` replaces `preset=6`. LZMA preset 9 uses maximum compression effort, typically saving 5–25% over preset 6 at the cost of slower compression. Compression runs once after training, not during evaluation, so this has no effect on timing.

## Inherited Stack (2026-03-24 AttnRes)

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
| Quantization | GPTQ-lite int6 + lzma-9 |
| Optimizer | Parameter Banking + Parallel Muon |
| AttnRes | Full (11 layers), zero-init |
| TTT | Legal score-first, 3ep, all blocks |

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
GATED_ATTENTION=1 VALUE_RESIDUAL=1 MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.2 \
TTT_ENABLED=1 TTT_USE_ADAM=1 TTT_ADAM_LR=0.0002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Ablation plan

To isolate each contribution, run with individual flags disabled:
- `GATED_ATTENTION=0` — ablate gated attention
- `VALUE_RESIDUAL=0` — ablate value residual
- `MTP_NUM_HEADS=0` — ablate MTP training signal
- `TTT_USE_ADAM=0` — revert to SGD TTT (compare adaptation quality)

## Credits

- **AttnRes**: Kimi Team paper (Attention_Residuals.pdf)
- **LeakyReLU²**: PR #493, PR #518
- **Parallel Muon**: PR #399
- **TTT recipe**: PR #461
- **Base model**: PR #414
