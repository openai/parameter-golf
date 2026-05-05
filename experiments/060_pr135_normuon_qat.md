# Experiment 060: PR135 + NorMuon + int6 QAT — NEW BEST 1.1474

## Status: COMPLETED

## Results
| Metric | Value |
|--------|-------|
| Steps | 7,331 @ 81.9ms/step |
| Artifact | 18,049,351 bytes ❌ |
| Standard eval | **1.1683 BPB** |
| **Sliding eval** | **1.1474 BPB** ← ALL-TIME BEST |
| Quant gap | ~0.002 (vs 0.004 without QAT) |

## KEY FINDING
int6 QAT adds ZERO step overhead (81.9ms = same as 058) but reduces quant gap by 0.002.
PR135 was wrong to disable QAT — their claim of "54% overhead" was for int8 QAT, not int6.
int6 QAT with CastedLinear + STE is essentially free.

## Progression from PR135 baseline
| Config | Sliding BPB | Improvement |
|--------|-------------|-------------|
| PR135 exact (057) | 1.1535 | baseline |
| + NorMuon (058) | 1.1494 | -0.0041 |
| **+ QAT (060)** | **1.1474** | **-0.0061** |

## wandb
- Run name: 060_pr135_normuon_qat
