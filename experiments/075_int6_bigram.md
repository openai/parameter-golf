# Experiment 075: Learned BigramHash + int6 quant — NEW BEST SUBMISSION-READY!

## Results
| Metric | Value |
|--------|-------|
| Sliding eval | **1.1448 BPB** ← ALL-TIME BEST THAT FITS! |
| Standard eval | 1.1659 |
| **FLAT+zstd** | **15,795,185 bytes ✅ (205KB under 16MB!)** |
| Model params | 22,368,841 |

## KEY BREAKTHROUGH
Classifying bigram.embed as "mlp" → int6 quantization → FITS under 16MB!
Previously the bigram embed was fp16 passthrough → 17.38MB (over budget).
Now int6 → 15.80MB ✅ with only ~0.001 BPB quality loss from int6 vs fp16.

## Comparison
| Config | Sliding | Artifact |
|--------|---------|----------|
| No bigram (071) | 1.1480 | 15.33MB ✅ |
| fp16 bigram (070) | 1.1458 | 17.38MB ❌ |
| **int6 bigram (075)** | **1.1448** | **15.80MB ✅** |

## This is our SUBMISSION CANDIDATE. Beats PR162 (1.1483). Close to PR179 (1.1472).
