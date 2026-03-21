# Experiment 084: leaky_relu(0.5)² — NEW BEST SUBMISSION-READY!

## Results
- **Sliding: 1.1427** ← NEW BEST!
- Standard: 1.1639
- FLAT+zstd: 15.76MB ✅
- Steps: 7,257 @ 82.7ms/step

## Activation comparison
| Activation | Sliding | Diff vs relu² |
|-----------|---------|---------------|
| relu² (081) | 1.1441 | baseline |
| abs² (083) | 1.1480 | +0.004 (worse) |
| **leaky_relu(0.5)² (084)** | **1.1427** | **-0.0014 (better!)** |

## Finding
leaky_relu(0.5)² outperforms relu² by 0.0014 BPP.
Softer gating (shrink negatives by 50% instead of zeroing) preserves more info.
The X post claim about leaky_relu was correct!
