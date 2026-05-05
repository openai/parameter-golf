# Experiment 091: PR198 tuned config + leaky2, 11L (clean script)

## Config
- **Script**: clean_train.py (PR198 base + our additions)
- 11 layers, model_dim=512, MLP 3x (h=1536)
- leaky_relu(0.5)² activation
- WD: MUON_WD=0.04, ADAM_WD=0.04 (PR198's config)
- LR: MATRIX_LR=0.025, SCALAR_LR=0.025, TIED_EMBED_LR=0.035
- SWA enabled (PR198 default), warmdown=3000
- BigramHash 4096x128, SmearGate, OrthoInit (all PR198 defaults)
- 8×H100, ~95.85ms/step

## Results
- Steps: 6,260 @ 95.85ms/step
- Pre-quant BPP: 1.1473
- SWA: averaged 8 checkpoints
- **Standard BPP: 1.1858**
- **Sliding BPP: 1.1634** ← WORSE than expected
- **Artifact: 12.25MB ✅** (3.75MB under budget!)
- Code size: 64.8KB

## Analysis
- Artifact size is EXCELLENT — 11L fits easily at 12.25MB with clean script
- But BPP is disappointing: 1.1634 vs PR198's claimed ~1.13
- The gap is likely from:
  1. WD=0.04 is not aggressive enough (andrewgcodes uses 0.08)
  2. SWA with WD=0.04 may not help (our earlier finding)
  3. ~6260 steps may not be enough for 11L to converge
  4. Missing some PR198 optimizations we haven't replicated

## Next: exp092 with WD=0.08 + int5 + pruning
