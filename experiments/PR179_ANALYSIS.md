# PR179 Analysis — NEW SOTA: 1.1472 mean BPB (3 seeds), 15.9MB ✅

## THIS IS NOW THE TARGET TO BEAT

## Key Innovations
1. **11 layers** (vs everyone else's 9) — more depth! WD makes int6 compress well enough to fit.
2. **WD=0.038** on Muon — highest WD we've seen. Our sweep only went to 0.04.
3. **LR=0.025** (vs our 0.02) — 25% higher matrix LR
4. **Plain Muon** (not NorMuon)
5. Int6 per-row + fp16 embed + zstd-22
6. Sliding window stride=64
7. ~73ms/step (Modal 8xH100), ~7000 steps

## Results (3 seeds)
| Seed | Sliding BPP | Artifact |
|------|-------------|----------|
| 1337 | 1.1482 | 15.9MB |
| **42** | **1.1443** | 15.9MB |
| 7 | 1.1492 | 15.9MB |
| **Mean** | **1.1472** | |

## What we should try
1. **11 LAYERS + our NorMuon + QAT + WD=0.038** — combine their depth with our optimizer
   - Need to reduce MLP to fit: 11 layers × MLP 3x (h=1536) = more params
   - OR keep MLP=1536 and rely on int6+zstd compression
   - At 11 layers, the model has ~25% more params than 9 layers
2. **LR=0.025** sweep on our config — easy test
3. **WD=0.038** on our config — easy test (we only went to 0.04)

## Experiment Plan
- **076**: 11 layers + NorMuon + QAT + WD=0.038 + LR=0.025 (PR179 config + our additions)
- **077**: Same but with MLP reduced if needed to fit
