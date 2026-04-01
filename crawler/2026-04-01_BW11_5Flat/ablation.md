# BW11_5Flat — Ablation Results

## Gate (4×GPU SDPA, 2000 steps, seed=444)

| Arm | raw BPB | int6_sw BPB | step_avg | bytes | params |
|-----|---------|-------------|----------|-------|--------|
| BWFF-00 (NUM_FLAT_LAYERS=4) | 1.3050 | 1.28756847 | 141.90ms | 9,542,577 (9.54MB) | 14,528,044 |
| BWFF-01 (NUM_FLAT_LAYERS=5) | 1.2918 | 1.27305177 | 158.53ms | 10,715,655 (10.72MB) | 16,889,396 |
| **delta** | **−0.0132** | **−0.01452** | +16.63ms | +1.18MB | +2.36M |

**Verdict: GATE PASS**

- int6_sw improvement: −0.01452 BPB ✓ (largest gate delta seen in this lab)
- Size under 16MB ✓
- Step time +16.6ms — expected cost of 2 extra flat layers (~11.7% slower)

---

## Full Run (8×H100, seed=444)

| metric | value |
|--------|-------|
| raw BPB | 1.1876 |
| int6_sw BPB | **1.17651313** |
| quant gap | 0.01109 |
| step_avg | ~85ms |
| steps | 7074 (wallclock stop) |
| bytes | 10,048,191 (10.05MB) |

**vs BW5 champion (1.18672385): −0.01021 BPB — NEW CHAMPION**
**vs BW10_GPTQ (1.18292670): −0.00641 BPB**

Gate proxy inflation: 1.42× (gate −0.01452, full run −0.01021 vs BW5). Clean.

### Quant gap note

5F with naive int6 (gap 0.01109) has a *smaller* quant gap than 4F+GPTQ (gap 0.01697).
Better model weights are inherently more quantization-friendly. BW12 (5F+GPTQ) targets
the remaining 0.011 gap.

---

## Confirmation Run (8×H100, seed=300)

| metric | value |
|--------|-------|
| raw BPB | 1.1874 |
| int6_sw BPB | **1.17490448** |
| quant gap | 0.01395 |
| step_avg | ~85ms |
| steps | 7077 (wallclock stop) |
| bytes | 10,343,385 (10.34MB) |

**Confirmed.** seed=300 beats champion (1.18672385) by −0.01182 BPB.
Mean across seeds 444+300: **1.1757 BPB**

---

## Verdict

**PROMOTES.** BW11_5Flat (BW8 + NUM_FLAT_LAYERS=5) is the new crawler champion at 1.17651313 BPB.

Next experiment: BW12 = BW11 base + LOOP_AWARE_GPTQ=1 (stack both levers).
