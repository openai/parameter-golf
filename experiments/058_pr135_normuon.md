# Experiment 058: PR135 + NorMuon — NEW BEST 1.1494 BPB

## Status: COMPLETED

## Results
| Metric | Value |
|--------|-------|
| Steps | 7,326 @ 81.9ms/step |
| Artifact | 17,851,054 bytes ❌ (1.85MB over) |
| Standard eval | **1.1704 BPB** |
| **Sliding eval s64** | **1.1494 BPB** ← ALL-TIME BEST |

## Comparison
| | PR135 (Muon) | 057 (reproduce) | **058 (NorMuon)** |
|--|-------------|-----------------|-------------------|
| Sliding | 1.1539 | 1.1535 | **1.1494** |
| Standard | 1.1748 | 1.1744 | **1.1704** |

## KEY FINDING
NorMuon improves PR135's script by 0.004 BPB. We now BEAT PR135.
The artifact compression issue remains (17.85MB vs 16MB cap).

## wandb
- Run ID: check dashboard
- Run name: 058_pr135_normuon
