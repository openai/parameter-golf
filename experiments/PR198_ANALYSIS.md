# PR198 Analysis — 1.1318 BPB sliding, 15.7MB ✅ (NEW HIGHEST)

## Key Techniques We Should Adopt

### 1. BIGRAM_VOCAB_SIZE=2048 (not 4096) — EASY WIN
Saves ~300KB artifact, "negligible BPB cost". We use 4096.
At 2048 buckets: nn.Embedding(2048,128) = 262K params vs 524K = halved.
**EXPERIMENT: Try BIGRAM_VOCAB_SIZE=2048 on our 9-layer config**

### 2. WD=0.04 on BOTH Muon AND AdamW
We apply MUON_WEIGHT_DECAY=0.03 but AdamW WD is still 0.01.
PR198 uses 0.04 on both.
**EXPERIMENT: Set WEIGHT_DECAY=0.04 (AdamW) + MUON_WEIGHT_DECAY=0.04**

### 3. SCALAR_LR=0.025, TIED_EMBED_LR=0.035
We use SCALAR_LR=0.02, TIED_EMBED_LR=0.03.
PR198's higher LRs might help convergence in fewer steps.
**EXPERIMENT: Try these LRs**

### 4. FA3 — they get 81ms/step for 11 layers
We get 99ms. FA3 saves ~18ms/step → ~1300 more steps for 11L.
**Still need to install FA3 on our machine**

### 5. SWA with ~8 checkpoints — works for them
Our SWA test (074) was neutral, but they use it successfully.
Maybe the combination with higher WD + SWA works on plain Muon but not NorMuon.

### 6. ITERATIONS=9000 cap
They explicitly cap at 9000 iterations. Not sure why — maybe prevents overfitting with wallclock-based warmdown?

## Artifact Size Gap
PR198: 26.8M params → 15.7MB on their platform
Ours 076: 27.1M params → 18.95MB on our platform
Gap: ~3.2MB for essentially same model. Platform difference is killing us.
MUST test on Runpod to see actual submission size.

## Updated experiment priorities
1. 079 (10 layers) — RUNNING NOW
2. Try BIGRAM_VOCAB_SIZE=2048 on best 9L config
3. Try WD=0.04 on both Muon and AdamW
4. Try higher SCALAR_LR=0.025, TIED_EMBED_LR=0.035
5. Test on Runpod ASAP
