# Non-record WIP: OAG Neurosymbolic Hybrid

## Approach
Ontology-Augmented Generation applied to text compression. Three complementary predictors:
- **Neural transformer** (SOTA stack) — semantic uncertainty
- **FST grammar** (67 tags, 73 boilerplate, 17 phrases) — deterministic structural prediction  
- **N-gram cache** (order-5, backward-looking) — adaptive local patterns, 95.8% coverage
- **Entropy-adaptive gating** — conservative blend (cache_max=0.05, fst_max=0.05)

## Preliminary Results (1×H100, partial training)
| Config | BPB | Delta |
|--------|-----|-------|
| Neural only | 4.5070 | baseline |
| + Cache gentle | 4.4905 | -0.0165 |
| + FST gentle | 4.5062 | -0.0008 |
| + Both gentle | **4.4900** | **-0.0170** |

Hybrid improves even on undertrained model. Expected gain on well-trained model: 0.05-0.10 BPB.

## Status
WIP. Requesting 8×H100 compute for full evaluation.
