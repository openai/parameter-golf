# Bandit_wagon_5f_ablations Results — 2026-03-30

**Setup:** seed=444, 500 steps, warmdown=0, SKIP_GPTQ=1, CRAWLER_QUANT_INT8=1, mlp_mult=6.0
**Note:** Pod missing zstandard — fell back to zlib (affects submission size only, NOT int6_sw_bpb)

## Results

| ARM | Config | XSA_LAST_N | Params | Raw val_bpb @500 | INT6_SW_BPB | Quant gap |
|-----|--------|:----------:|-------:|:----------------:|:-----------:|:---------:|
| BW-03 (ref) | 5F+1C | 11 | 16,823,860 | 1.4254 | 1.54404 | 0.1186 |
| BW2-00 | **4F+1C** | 11 | 14,462,508 | 1.4250 | **1.52365** | **0.0987** |
| BW2-01 | 5F+1C | 14 | 16,823,860 | 1.4239 | 1.52963 | 0.1057 |

## Key Finding

**4F+1C wins.** BW2-00 beats BW-03 (the 5F+1C proxy that appeared to win) by 0.020 BPB
when given a proper control at equal compute.

Raw learning rate is identical across all three arms (~1.424 raw val_bpb). The entire
difference lives in quantization robustness:

- 4F+1C: quant gap = 0.099
- 5F+1C + XSA=14: quant gap = 0.106
- 5F+1C + XSA=11: quant gap = 0.119

Fewer parameters = less quantization sensitivity. 5F+1C adds ~2.4M params which hurt
post-quant BPB even though they don't hurt raw loss.

## Secondary Finding: XSA Coverage Is a Quantization Robustness Lever

BW2-01 (XSA=14) recovered 0.015 BPB vs BW-03 (XSA=11) for the same 5F+1C model.
Increasing XSA coverage from 61% → 78% cut the quantization gap by ~11%. This suggests
XSA acts as a regularizer that improves quantization robustness in deeper models.

## Decision (per HYPOTHESIS.md rules)

> BW2-00 (1.52365) < BW-03 (1.54404) → 4F wins → STOP. Do not book 8×H100 for 5F.

**Verdict: 4F+1C is optimal at this parameter budget. BW-03's apparent win was an
artifact of not having a proxy control arm. CL3 config is confirmed correct.**

## Open Thread

XSA coverage vs quant gap is worth one more probe: does XSA=12 or XSA=13 on the 4F+1C
baseline (currently XSA=11, 73% coverage) improve the full-run score? Small lever, cheap
to test, and the mechanism is now understood.
