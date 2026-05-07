# Parameter Golf Submission (DualHash AdaMuon MoE)

**Author:** Karen Poghosyan (@Karen042009)

## Performance Summary
- **Track:** 10 Minute Training / 16MB Budget
- **FineWeb BPB:** 0.83353253 (Mean of 3 seeds: 0.83386159, 0.83415485, 0.83258115)
- **Artifact Size:** ~15.13 MiB avg (LZMA compressed)
- **Parameters:** 22,070,160
- **Training Time:** ~9.9 minutes on 8xH100 (~7100 steps / seed)
- **Data Setup:** Initial data download and pre-tokenization takes ~25+ minutes.

### Per-Seed Results
| Seed | FineWeb BPB | Steps | Size (LZMA) | Train Time |
|------|-------------|-------|-------------|------------|
| 1337 | 0.83386159  | 6700  | 15.15 MB    | 562.25s (9m 22s) |
| 42   | 0.83415485  | 7100  | 15.10 MB    | 595.96s (9m 55s) |
| 2025 | 0.83258115  | 7100  | 15.14 MB    | 594.74s (9m 54s) |

## Architectural Details
This model implements several state-of-the-art optimizations for the 16MB/10min track:

1.  **DualTokenHashSkip:** Utilizes two separate hash tables (2×2048×16) with prime multipliers (8191 and 104729) to create a robust bigram skip connection that is concatenated before being projected into the hidden dimension.
2.  **LayerScale Recurrence:** Implements a stable recurrent structure [0, 1, 2, 3, 4, 5, 3, 4, 5] using learnable LayerScale coefficients (initialized at 1.0 for the main branch and 0.1 for the recurrent branch) to manage gradient flow through deep loops.
3.  **SharedMoE (Mixture of Experts):** A hybrid MoE architecture featuring 1 shared expert and 3 specialized experts. The router selects the top-1 expert, and its output is combined with the shared expert, providing high capacity with sparse computation.
4.  **AdaMuon Optimizer:** An advanced variant of the Muon optimizer that incorporates RMS pre-conditioning and Riemannian Newton-Schulz orthogonalization for faster and more stable convergence.
5.  **Dynamic MSE SDClip:** A smart quantization guard that searches over a grid of standard deviation clipping thresholds (σ ∈ {2.5, 3.0, 3.5, 4.0}) during export to find the optimal INT6 representation that minimizes Mean Squared Error.
6.  **Score-First TTT:** Test-Time Training with a strictly compliant 2-pass implementation (evaluating the score before adapting the model state) to ensure legal competition performance.

## Files
- `train_gpt.py`: Full training script.
- `train.log`: Full logs containing the complete training and execution process.
- `submission.json`: Competition metadata.
- `README.md`: Submission documentation.
