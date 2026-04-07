# David Ghazaryan — MoE + BigramHash4096

## Results
| Seed | val_bpb | Size (bytes) |
|------|---------|--------------|
| 1337 | 1.11764880 | 15,873,596 |
| 42   | 1.11891002 | 15,893,104 |
| 2025 | 1.11742168 | 15,908,116 |
| **mean** | **1.11799350** | |

## Novel Contributions
1. **BigramHash4096** — expanded bigram vocabulary from 3072 to 4096
2. **MoE MLP** — Mixture of Experts in the MLP layers (first explored in this repo)

## Hardware
8× H100 80GB (YSU HPC Cluster)
