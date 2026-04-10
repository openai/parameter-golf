# SP4096 + 3-Layer Recurrence + GPTQ Embeddings + SDClip + ETLB

**val_bpb = 1.0913** (3-seed mean, std 0.0012) | **~14.75 MB** | 8xH100 SXM

## Results

| Seed | Sliding BPP | ETLB BPP | Artifact |
|------|-------------|----------|----------|
| 42   | 1.0901      | 1.0900   | 14,748,056 |
| 314  | 1.0920      | 1.0919   | 14,744,841 |
| 999  | 1.0922      | 1.0921   | 14,746,245 |
| **Mean** | **1.0914** | **1.0913** | **14,746,381** |

## Key Techniques

1. **SP4096 Vocabulary** — 4096-token SentencePiece BPE
2. **GPTQ on Embeddings** (int8) — PR #1394 innovation, saves ~2MB artifact vs FP16 embeddings
3. **SDClip** — std-dev based quantization clip thresholds (PR #1394)
4. **3-Layer Depth Recurrence** (layers 3,4,5, from step ~2950) — extends PR #1204
5. **QK-Gain 5.0** — PR #1217 @bigbag
6. **WD=0.095 + MLR=0.022** — higher WD for compression, higher LR to compensate (PR #1331)
7. **ETLB: Eval-Time Logit Bias** — optimizes vocab bias vector during sliding window eval (PR #1399)
8. **MuonEq-R** — row-normalize before Newton-Schulz (PR #1260)
9. **Full GPTQ int6 + Brotli** compression
10. **LZMA code wrapper** — minified code saves ~40KB artifact

## ETLB Details

During sliding window evaluation, a bias vector `b` in R^4096 is optimized on context tokens:
- SGD, 5 steps per window, lr=0.05
- Warm-started across windows (carries document-level token preferences)
- Clipped to [-3, 3]
- Applied to logits after model forward pass
- No gradients through model — only the bias vector is updated

## Compliance

- No SLOT — no eval-time hidden state adaptation
- No TTT — no model weight updates during evaluation
- ETLB only modifies a separate bias vector, not model weights
- GPTQ calibration within training budget
- Standard autoregressive sliding-window eval (stride=64)
- All artifacts under 16,000,000 bytes

## Reproduce

```bash
pip install brotli
VOCAB_SIZE=4096 QK_GAIN_INIT=5.0 MUON_WD=0.095 MATRIX_LR=0.022 \
  LOOP_START=3 LOOP_END=5 \
  ETLB_ENABLED=1 ETLB_LR=0.05 ETLB_STEPS=5 ETLB_CLIP=3.0 \
  SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- PR #1394 @clarkkev (GPTQ embeddings + SDClip base)
- PR #1399 @resouer (ETLB concept)
- PR #1331 @dexhunter (3-layer recurrence + WD-LR synergy)
- PR #1217 @bigbag (QK-Gain 5.0)
- PR #1260 @dexhunter (MuonEq-R)
- PR #1204 @msisovic (depth recurrence)
