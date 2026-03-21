# PR162 Analysis — mean val_bpb=1.1483 (3 seeds), 15.92MB ✅

## THIS IS THE NEW TARGET TO BEAT

## Key Techniques
Same base as PR135 but with TWO important additions:
1. **Muon WD=0.02** (weight decay on Muon optimizer, not just AdamW)
2. **SWA** over last 50% of training, every 200 steps

## Config
- 9 layers, dim=512, MLP 3x (h=1536), seq2048, batch=786K
- OrthoInit + SmearGate + BigramHash
- Plain Muon (NOT NorMuon), momentum 0.99
- matrix_lr=0.02, grad_clip=0.3
- Muon weight_decay=0.02, AdamW weight_decay=0.01
- SWA: last 50% training, snapshot every 200 steps
- int6 per-row + fp16 embed + last-layer K fp16
- zstd-22, sliding eval stride=64

## Results (3 seeds)
| Seed | Sliding BPB |
|------|-------------|
| 1337 | 1.1489 |
| 42 | 1.1485 |
| 7 | 1.1476 |
| **Mean** | **1.1483** |

## What we should try from PR162

### 1. Muon WD=0.02 — HIGH PRIORITY
We use weight_decay=0.01 on AdamW but NOT on Muon (NorMuon).
PR162 applies WD=0.02 directly to Muon. This is DIFFERENT.
Muon weight decay regularizes the matrix params → tighter weight distribution → better compression AND quality.
**EXPERIMENT: Add weight_decay to NorMuon optimizer**

### 2. SWA with their settings — MEDIUM PRIORITY (contradicts our findings)
We tried LAWA/SWA before and it HURT. But PR162 gets 1.1483 WITH SWA.
Differences: they use plain Muon + WD=0.02 + SWA start at 50% warmdown.
Our failures were with NorMuon + no WD + SWA start at various points.
Maybe SWA works specifically with Muon WD=0.02.
**EXPERIMENT: Try SWA with Muon WD=0.02 on our config**

### 3. Plain Muon + WD=0.02 (instead of NorMuon + WD=0.01) — WORTH RETESTING
PR162 uses plain Muon with WD. Our exp052 tested plain Muon WITHOUT WD and it was worse.
Maybe plain Muon + WD=0.02 is the winning combo.
**EXPERIMENT: Test plain Muon + WD=0.02**

### 4. Their artifact is 15.92MB — fits on their platform with BigramHash!
Same model as ours but fits. Platform difference explains it.
We should test our FLAT+zstd WITH BigramHash — it might fit under 16MB now.

## Updated Experiment Queue
- **069**: Add WEIGHT_DECAY to NorMuon + our best config (test Muon WD=0.02 effect)
- **070**: Plain Muon + WD=0.02 + SWA (replicate PR162's exact approach + our QAT)
- **071**: Test BigramHash WITH FLAT+zstd serialization (does it fit under 16MB?)
