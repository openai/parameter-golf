# 11L EMA + AdamW TTT (10ep) — val_bpb: 1.1027

**Author:** Sungjoon Park (@sjp611)
**Track:** 10min / 16MB
**Hardware:** 8xH100 SXM, 600s training
**Artifact:** ~15.75 MB (int6+zstd)

## Approach

Built on PR #398 (11L EMA + SGD TTT 20ep, 1.1213 BPB) with a single key change:

**Replace SGD with AdamW for test-time training.**

- SGD(lr=0.008, momentum=0.9) for 20 epochs → AdamW(lr=0.0005, wd=0.0) for 10 epochs
- AdamW achieves significantly more loss reduction per epoch than SGD
- TTT time reduced from ~260s to ~157s (40% faster)
- BPB improved by -0.019 vs prior SOTA

All other settings identical to PR #398:
- 11 layers, 512 dim, 8 heads / 4 KV heads (GQA)
- MLP 3x (hidden=1536), relu-squared activation
- SmearGate + BigramHash(2048, dim=128) + OrthoInit
- Partial RoPE (16/64 dims), LN Scale
- EMA(0.997), no SWA, no XSA, no Late QAT
- Int6 mixed quantization + zstd-22 compression
- Sliding window evaluation (stride=64)

## Code diff from PR #398

```diff
-    ttt_lr = float(os.environ.get("TTT_LR", 0.008))
-    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 20))
+    ttt_lr = float(os.environ.get("TTT_LR", 0.0005))
+    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 10))

-    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
+    optimizer = torch.optim.AdamW(ttt_params, lr=args.ttt_lr, weight_decay=0.0)
```

## Results (3-seed, sliding window stride=64)

| Seed | Steps | val_loss | val_bpb |
|------|-------|----------|---------|
| 1337 | 4372 | 1.8675 | 1.1060 |
| 42 | 4578 | 1.8560 | 1.0992 |
| 7 | 4612 | 1.8623 | 1.1030 |
| **Mean** | | **1.8620** | **1.1027** |
| **Std** | | | **0.0034** |

## Comparison to prior SOTA (PR #398)

| Metric | PR #398 (SGD TTT) | Ours (AdamW TTT) |
|--------|-------------------|-------------------|
| Best BPB | 1.1213 | 1.0992 |
| Mean BPB | 1.1221 | 1.1027 |
| TTT epochs | 20 | 10 |
| TTT time | ~260s | ~157s |
| Improvement | — | -0.0194 |

## Run command

```bash
SEED=1337 NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=0 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=0 \
TTT_ENABLED=1 TTT_LR=0.0005 TTT_EPOCHS=10 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
