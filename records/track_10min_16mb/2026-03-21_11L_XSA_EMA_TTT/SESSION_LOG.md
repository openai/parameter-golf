# Session Log — 2026-03-21

val_bpb: 1.1708 → 1.1518 | 8×H100 SXM | 600s wallclock

## PR history

| PR | val_bpb | Status | Takeaway |
|----|---------|--------|----------|
| #145 | 1.2052 | Closed | Int8 QAT overhead exceeds benefit at 10-min wallclock |
| #174 | 1.1933 | Closed | SWA needs 50+ checkpoints; doc-isolated eval regresses at stride=64 |
| #189 | 1.1929 | Closed | Stacking unvalidated features doesn't compound — isolate first |
| #212 | 1.1708 | Closed | Int6 + 3x MLP is the single biggest lever. 5-seed validated (mean 1.1724, std 0.0026) |
| — | **1.1518** | Active | Implementation fixes + U-Net skips + corrected hyperparameters |

## Result progression

| Run | val_bpb | Steps | ms/step | Changes |
|-----|---------|-------|---------|---------|
| 9L Int6 3xMLP | 1.1708 | ~12,500 | 48 | Baseline: 9L, seq=1024, int6, 3xMLP |
| 11L + XSA + EMA + TTT | 1.1716 | 7,081 | 81 | Added XSA, EMA, TTT, tuned HPs |
| 11L + implementation fixes | **1.1518** | ~7,100 | 82 | Fixed SmearGate, BigramHash, U-Net skips, ROPE_BASE |
| 11L + Partial RoPE + LN Scale | TBD | — | — | In progress |

## Implementation fixes (1.1716 → 1.1518)

**SmearGate**: Changed from additive (`x + g*prev`) to interpolation (`torch.lerp(x, prev, g)`). The additive formula inflated magnitude by up to 1.5x at default gate values.

**BigramHash**: Replaced polynomial hash with XOR-based hash using larger primes for better bucket distribution. Added a learned output scalar (init 0.05) that was missing.

**U-Net skip connections**: Were defaulting to off. Enabled encoder/decoder structure (5/6 split for 11L) with learned per-channel skip weights. Restructured as single `nn.Parameter` tensor and two clean loops for torch.compile compatibility.

**ROPE_BASE**: Corrected from 50000 to 10000.

**Optimizer coverage** (found in prior session): SmearGate and BigramHash parameters were not in any optimizer group — both modules were frozen from initialization. Fixed by adding them to the appropriate optimizer groups.

## Revised findings from PR #212 ablations

PR #212 included 9 controlled ablations. One key conclusion has been revised:

**SmearGate + BigramHash** originally showed a +0.003 BPB regression on int6. This was accurate at the time, but the measurement reflected four implementation issues (listed above) that prevented both features from training properly. With corrections applied, both features contribute positively.

## Approaches that didn't improve results

**NTK-RoPE** (train at seq=1024, eval at 2048): Halving sequence length doubles the batch sequence count at constant `TRAIN_BATCH_TOKENS`, so per-step compute stays roughly the same. No meaningful speedup observed.

**TRAIN_BATCH_TOKENS=786432**: Added 20-30ms/step overhead on our hardware (101-116ms vs 82ms at 524K), reducing total steps from ~7300 to ~5200. The gradient quality improvement did not offset the step count loss.

**Step-based QAT activation**: Triggering at a fixed fraction of `ITERATIONS` (which is set high for the wallclock cap) caused QAT to activate too late with insufficient steps remaining. Replaced with an lr_scale-based trigger that activates during the last ~10% of warmdown.

## Performance observations

Step time creeps from ~70ms to ~82ms during training, likely due to per-step EMA state dict iteration. Committed but not yet validated: cached parameter references for EMA, pre-allocated Muon flat buffer, fused SmearGate kernel via `torch.lerp`.

## Config (1.1518)

```
NUM_LAYERS=11  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4  MLP_MULT=3
QUANT_BITS=6  USE_ZSTD=1  TIE_EMBEDDINGS=1  FP16_EMBED_EXPORT=1
TRAIN_SEQ_LEN=2048  EVAL_SEQ_LEN=2048  ROPE_BASE=10000
TRAIN_BATCH_TOKENS=524288  WARMDOWN_ITERS=3000
XSA_LAST_N=4  EMA_ENABLED=1  EMA_DECAY=0.997
TTT_ENABLED=1  TTT_LR=0.002  TTT_EPOCHS=3  TTT_FREEZE_BLOCKS=2
SMEAR_GATE=1  BIGRAM_HASH=1  BIGRAM_HASH_BUCKETS=2048
ORTHO_INIT=1  UNET_SKIPS=1
MUON_MOMENTUM=0.99  GRAD_CLIP_NORM=0.3  MUON_WD=0.04  ADAM_WD=0.04
MATRIX_LR=0.025  SCALAR_LR=0.025  TIED_EMBED_LR=0.035
DOC_ISOLATED_EVAL=1  EVAL_STRIDE=64
QAT=1 (lr_scale < 0.1, absmax int6 STE)
```
