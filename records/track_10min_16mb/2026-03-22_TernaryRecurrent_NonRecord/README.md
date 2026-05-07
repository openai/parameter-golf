# Non-record: TernaryRecurrentGPT + Depth Recurrence (1xL4 local validation)

## Summary

- 7 unique layers x 2 loops = 14 effective depth
- Ternary MLP weights (1.58-bit STE QAT)
- BigramHash 2048
- SmearGate
- Muon+AdamW WD=0.04
- SWA from 40%

## Results

| Hardware | Steps | val_bpb (post-quant) | Artifact |
| --- | --- | --- | --- |
| 1xL4 24GB | 20,000 | 1.5348 | 12,372,468 bytes |

Local validation run at reduced batch (32k tokens, seq=512). Full 8xH100 competition-scale run pending compute grant.

## Key findings

- ternary quant gap only 0.002 bpb (pre=2.5044, post=1.5348)

## Negative findings

- Neural Cache adds +0.028 bpb at this scale, disabled for now
