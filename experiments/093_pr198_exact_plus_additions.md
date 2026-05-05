# Experiment 093: PR198 EXACT config + leaky2 + int5 + prune

## Config (PR198 exact + 3 additions)
- PR198 exact: NUM_LAYERS=11, BIGRAM_VOCAB_SIZE=2048, MUON_WD=0.04, ADAM_WD=0.04
- PR198 exact: MATRIX_LR=0.025, SCALAR_LR=0.025, TIED_EMBED_LR=0.035
- PR198 exact: MUON_MOMENTUM=0.99, WARMUP_START=0.92, WARMUP_STEPS=1500
- PR198 exact: WARMDOWN_ITERS=3000, ITERATIONS=9000, EVAL_STRIDE=64
- **OUR ADDITIONS**: MLP_ACTIVATION=leaky2, INT5_MLP=1, PRUNE_FRAC=0.02
- Script: clean_train.py, 8×H100, ~96ms/step

## Results
- Steps: 6,261 @ 95.85ms/step (vs PR198's 7,412 @ 81ms — no FA3!)
- SWA: averaged 8 checkpoints
- **Standard BPP: 1.1777**
- **Sliding BPP: 1.1549** ← WORSE than PR198's 1.1318 by 0.023!
- **Artifact: 13.71MB ✅** (2.29MB under budget)

## Analysis
- BPP gap vs PR198 (0.023): probably from ~1150 fewer steps (no FA3) + int5 post-quant BPP cost
- Artifact 13.71MB vs PR198's 15.7MB: int5 MLP + pruning + our code saved ~2MB
- Need exp094 (PR198 exact NO additions) to isolate our additions' effect

## PR198 comparison
| | PR198 (their machine) | Exp093 (our machine) | Delta |
|---|---|---|---|
| Sliding BPP | 1.1318 | 1.1549 | +0.023 worse |
| Artifact | 15.7MB | 13.71MB | -2.0MB better |
| Steps | 7,412 | 6,261 | -1,151 fewer |
| Step time | 81ms | 96ms | +15ms (no FA3) |
