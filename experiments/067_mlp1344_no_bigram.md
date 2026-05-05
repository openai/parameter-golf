# Experiment 067: MLP=1344, no BigramHash — BUDGET FIT TEST

## Status: COMPLETED

## Results
| Metric | Value |
|--------|-------|
| Steps | 7,592 @ 79.2ms/step |
| Model params | 20,009,544 |
| Standard eval | 1.1799 BPB |
| **Sliding eval** | **1.1590 BPB** |
| **FLAT+zstd artifact** | **14,182,067 bytes ✅ (1.82MB under 16MB!)** |

## Compression comparison (this run)
| Format | Size |
|--------|------|
| torch.save + zstd | 14,422,311 |
| manual + zstd | 14,414,483 |
| manual + LZMA | 14,289,966 |
| **FLAT + zstd** | **14,182,067** ← best |

## Key Finding
Fits under 16MB with 1.82MB headroom. But BPB is 0.012 worse than 060 (1.1590 vs 1.1474).
Over-trimmed — 1.82MB headroom means we can increase MLP_HIDDEN back.
MLP=1536 + no BigramHash should fit ~15.4MB and recover most BPB.
→ Launching exp068 with MLP=1536, no BigramHash.
