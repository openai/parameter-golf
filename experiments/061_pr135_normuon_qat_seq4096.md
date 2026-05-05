# Experiment 061: PR135 + NorMuon + QAT + seq4096

## Status: COMPLETED

## Results
| Metric | Value |
|--------|-------|
| Steps | 10,739 @ 55.9ms/step |
| Artifact | 17,898,474 bytes ❌ |
| Standard eval | **1.1634** (better than 060's 1.1683) |
| Sliding eval | 1.1518 (worse than 060's 1.1474) |

## Comparison: seq2048 vs seq4096 (both with NorMuon + QAT)
| | 060 (seq2048) | 061 (seq4096) |
|--|--------------|--------------|
| Steps | 7,331 | 10,739 (+46%) |
| Standard | 1.1683 | **1.1634** (-0.005) |
| **Sliding** | **1.1474** | 1.1518 (+0.004) |

## KEY FINDING
seq4096 gets better standard eval (more steps) but WORSE sliding eval.
seq2048 models benefit MORE from sliding window context boost (+0.021 BPB improvement vs +0.012 for seq4096).

**CONCLUSION: Stick with seq2048 for sliding eval submission. 060 remains best at 1.1474.**

## wandb
- Run name: 061_pr135_normuon_qat_seq4096
