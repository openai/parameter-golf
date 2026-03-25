# 11L EMA + XSA + TTT + Int6 MLP3x (pending compute)

## Approach

Builds on PR #287 (jfprincz, val_bpb=1.1271, 15.5MB) by adding test-time training (TTT). TTT runs during eval only so artifact size is unchanged.

### What's new over #287
- **TTT**: Full-weight SGD on val data before scoring (3 epochs, lr=0.002, momentum=0.9)
- Freezes first 2 blocks during TTT for stability
- Full DDP support

### Inherited from #287
- 11 layers, 512-dim, 8 heads, 4 KV heads (GQA)
- 3x MLP (1536 hidden), relu-squared
- EMA (decay=0.997) replacing SWA
- XSA on last 4 layers
- Int6 per-row quant + zstd-22
- SmearGate + BigramHash (2048)
- Muon WD=0.04, momentum=0.99
- FlashAttention, OrthoInit, muP

### Expected improvement
TTT has shown ~0.003-0.005 bpb improvement in PRs #254, #264, #281. Applied to #287's 1.1271 base, projected score: ~1.122-1.124.

## Checklist
- [x] Submission folder
- [x] `README.md`, `submission.json`, `train_gpt.py`
- [ ] Training log (pending compute)
- [ ] BPB score (pending compute)
