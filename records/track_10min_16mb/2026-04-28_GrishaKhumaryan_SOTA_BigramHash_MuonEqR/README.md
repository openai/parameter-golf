# Parameter Golf Submission

**Author:** GrishaKhumaryan

## Performance Summary
- **Track:** 10 Minute Training / 16MB Budget
- **FineWeb BPB:** 0.94182151 (Mean of 3 seeds: 0.93714852, 0.94606347, 0.94225255)
- **Artifact Size:** ~15.83 MiB (LZMA compressed)
- **Parameters:** 22,694,928
- **Training Time:** ~10.0 minutes on 8xH100 (~4880 steps / seed)
- **Data Setup:** Initial data download and pre-tokenization takes ~25+ minutes.

### Per-Seed Results
| Seed | FineWeb BPB | Steps | Train Time |
|------|-------------|-------|------------|
| 1337 | 0.93714852  | 4880  | 599.69s (9m 59s) |
| 42   | 0.94606347  | 4790  | 599.20s (9m 59s) |
| 2025 | 0.94225255  | 4850  | 599.60s (9m 59s) |

## Architectural Details
This model incorporates several optimizations to maximize parameter efficiency and learning speed:

1.  **BigramHash Skip Connections:** A custom skip connection that injects bigram hash embeddings (4096 vocab, 32 dim) directly into the residual stream, helping the model capture local patterns without increasing the core hidden size.
2.  **3-Layer Progressive Recurrence:** A 6-layer physical model where the last 3 layers are recurred in a [0, 1, 2, 3, 4, 5, 3, 4, 5] pattern, effectively providing 9 layers of depth with only 6 layers' worth of parameters.
3.  **MuonEq-R Optimizer:** A Riemannian equivariant adaptation of the Muon optimizer, providing stable and scale-invariant weight updates.
4.  **SDClip (Sigma-Delta Quantization):** A custom 6-bit quantization scheme using error diffusion (sigma-delta) during the saving process, allowing us to fit a larger `hidden_size=464` within the 16MB budget.
5.  **Legal Score-First TTT:** Test-Time Training implemented with a strict 2-pass approach (score before adaptation) to ensure full compliance with the competition's evaluation rules.
6.  **Parallel Residuals & EMA:** Utilizing parallel blocks and Exponential Moving Average for smoother convergence and better validation performance.

## Files
- `train_gpt.py`: Full training script.
- `train.log`: Full logs containing the complete training and execution process.
- `submission.json`: Competition metadata.
- `README.md`: Submission documentation.