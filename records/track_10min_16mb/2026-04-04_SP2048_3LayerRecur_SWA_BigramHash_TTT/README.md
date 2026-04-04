# SP2048 + 3-Layer Recurrence + SWA + BigramHash + Legal TTT

**val_bpb = 1.0955** (3-seed mean, std 0.0004) | **~15.46 MB** | 8xH100 SXM

## Results

| Seed | Sliding BPB | TTT BPB | Artifact |
|------|-------------|---------|----------|
| 42   | 1.0965      | 1.0952  | 15,410,102 |
| 314  | 1.0972      | 1.0960  | 15,493,880 |
| 999  | 1.0967      | 1.0954  | 15,474,490 |
| **Mean** | **1.0968** | **1.0955** | **15,459,491** |

## Key Techniques

1. **SP2048 Vocabulary** — 2048-token SentencePiece BPE (2.89 bytes/token)
2. **3-Layer Depth Recurrence** (layers 3,4,5, start step 3000) — extends PR #1204/#1331
3. **Stochastic Weight Averaging** (from frac=0.75) — averaged ~1200 checkpoints
4. **BigramHash Embeddings** (vocab=2048, dim=128) — n-gram side channel added to logits
5. **Legal Score-First TTT** (SGD, lr=0.002, 3 epochs) — from PR #1326
6. **Parallel Residuals** (from layer 7) — PR #1204
7. **MuonEq-R + QK-Gain 5.0** — PR #1260, PR #1217
8. **WD=0.095 + MLR=0.022** — higher WD for compression with compensating LR (PR #1331)
9. **Full GPTQ int6 + Brotli** compression

## Compliance

- Legal score-first TTT (tokens scored before weight updates)
- No SLOT, no n-gram cache
- Training: 590s on 8xH100 SXM
- Eval (sliding + TTT): ~500s, within 600s budget
- All artifacts under 16,000,000 bytes

## Reproduce

```bash
pip install brotli
VOCAB_SIZE=2048 QK_GAIN_INIT=5.0 MIN_LR=0.05 \
  RECUR_LAYERS=3,4,5 RECUR_START_STEP=3000 PARALLEL_START_LAYER=7 \
  MUON_WD=0.095 MATRIX_LR=0.022 \
  TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 \
  SWA_ENABLED=1 SWA_START_FRAC=0.75 \
  BIGRAM_ENABLED=1 BIGRAM_VOCAB=2048 BIGRAM_DIM=128 \
  SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- PR #1326 @aryanbhosale (base code + TTT)
- PR #1218 @clarkkev (SP4096 base)
- PR #1204 @msisovic (depth recurrence + parallel residuals)
- PR #1331 @dexhunter (3-layer recurrence + WD-LR synergy)
- PR #1260 @dexhunter (MuonEq-R)
- PR #1217 @bigbag (QK-Gain 5.0)
