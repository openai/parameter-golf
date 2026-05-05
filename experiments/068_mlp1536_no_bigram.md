# Experiment 068: MLP=1536, no BigramHash, NorMuon+QAT — SUBMISSION CANDIDATE

## Status: COMPLETED

## Results
| Metric | Value |
|--------|-------|
| Steps | 7,389 @ 81.4ms/step |
| Model params | 21,779,016 |
| Standard eval | 1.1722 BPB |
| **Sliding eval** | **1.1513 BPB** |
| **FLAT+zstd artifact** | **15,326,792 bytes ✅ (673KB under 16MB!)** |

## All compression formats
| Format | Size | Fits? |
|--------|------|-------|
| torch.save+zstd | 15.71MB | ✅ |
| manual+zstd | 15.69MB | ✅ |
| manual+LZMA | 15.48MB | ✅ |
| **FLAT+zstd** | **15.33MB** | **✅** |

## FIRST SUBMISSION-READY RUN!
1.1513 sliding BPB beats PR135 (1.1539) and fits under 16MB.
Only 0.004 behind our best BPB (1.1474 from exp060).
Dropping BigramHash costs only ~0.004 BPB but saves ~1.5MB.
