# Experiment 065: All Compression Methods Combined

## Status: COMPLETED (partial — LZMA crashed, no sliding eval)

## Compression Results
| Format | Size | vs torch.save baseline |
|--------|------|----------------------|
| torch.save + zstd | 18,552,839 | baseline |
| **manual + zstd (outlier split)** | **17,764,306** | **-789KB (4.3%)** |
| LZMA | CRASHED (not installed) | — |
| FLAT tensors | CRASHED (LZMA dependency) | — |

## Key Finding
Outlier splitting (52 tensors, 21,012 total outliers) + manual serialization saves 789KB.
This is better than manual alone (340KB in exp063) — outliers make residual more compressible.
But still 1.76MB over 16MB budget.

## Next: Need to test LZMA (fix import) and FLAT tensors on working platform.
Also trying ROPE_BASE=200000 from andrewgcodes in exp066 for BPB improvement.
