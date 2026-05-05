# Experiment 057: PR135 Exact Script Reproduction

## Status: COMPLETED

## Results
| Metric | PR135 (theirs) | 057 (ours) |
|--------|---------------|-----------|
| Steps | 7,201 @ 83ms | 7,376 @ 81.4ms |
| Standard eval | 1.1748 | 1.1744 |
| **Sliding eval** | **1.1539** | **1.1535** |
| Artifact | 15.16MB ✅ | 17.76MB ❌ |
| Params | 22,368,841 | 22,368,841 |

## KEY FINDING
**BPB REPRODUCED AND SLIGHTLY BEATEN: 1.1535 vs 1.1539!**
The model quality matches — our hardware produces equivalent results.

**BUT artifact is 2.6MB larger** (17.76 vs 15.16MB). Same zstd level 22, same quantization code.
Possible causes: different zstd library build, different weight distributions from hardware RNG,
or torch.save format differences between PyTorch versions.

## Implications
1. PR135's approach WORKS on our hardware for BPB
2. The artifact compression issue needs to be solved separately
3. We should use PR135's script as our base going forward and improve on it
4. Our merged script (054/055) was ~0.008 behind — the gap was in code details (weight_decay, quantization)

## wandb
- Run name: 057_pr135_exact (no wandb in PR135's script)
