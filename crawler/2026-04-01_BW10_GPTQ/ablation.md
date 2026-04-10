# BW10_GPTQ — Ablation Results

## Gate (4×GPU SDPA, 2000 steps, seed=444)

| Arm | raw BPB | int6_sw BPB | step_avg | bytes |
|-----|---------|-------------|----------|-------|
| BWGQ-00 (SKIP_GPTQ=1, naive) | 1.3051 | 1.28889181 | 142.40ms | 9,237,284 (9.24MB) |
| BWGQ-01 (LOOP_AWARE_GPTQ=1) | 1.3041 | 1.28403477 | 142.33ms | 10,230,472 (10.23MB) |
| **delta** | −0.0010 | **−0.00486** | −0.07ms | +993,188 (+0.99MB) |

**Verdict: GATE PASS**

- int6_sw improvement: −0.00486 BPB ✓
- Step time identical (GPTQ is cleanly post-training, not in training loop) ✓
- Both artifacts under 16MB ✓
- GPTQ calibration: 38 layers in 5.3s (24 flat GPTQ, 10 crawler Hessians from phase2) ✓

### Notes

**Size increased +0.99MB (unexpected).**
GPTQ optimizes rows for perplexity, producing more varied int6 values. Naive rounding
clusters weights at repeating patterns → zstd exploits regularity. GPTQ breaks those
patterns → zstd pays more. Quality gain is real; compression density trades off against it.

**CL2 reference was −0.0062 BPB** at gate scale on the ClownCar baseline. This −0.0049
is in the expected range given different base model.

**Quant gap asymmetry:** gate gap ~0.001 BPB; full run gap ~0.012 BPB. GPTQ addresses
the full gap post-training. If delta scales with gap, full run improvement could be
5-10× larger than gate signal: −0.025 to −0.050 BPB.

---

## Full Run (8×H100, seed=444)

| metric | value |
|--------|-------|
| raw BPB | 1.1999 |
| int6_sw BPB | **1.18292670** |
| quant gap | 0.01697 |
| step_avg | ~76ms |
| steps | 7893 (wallclock stop) |
| bytes | 9,963,860 (9.96MB) |
| GPTQ calibration | 8.6s post-training |

**vs BW5 champion (1.18672385): −0.00380 BPB — NEW CHAMPION**

Gate delta was −0.00486; full run delivered −0.00380 vs champion. Proxy inflation ~1.3×.
GPTQ calibration 8.6s post-training, zero impact on training step count.
Size +0.72MB vs BW5 (9.96MB vs ~9.24MB), well under 16MB.

---

## Confirmation Run (8×H100, seed=300)

_Pending._

---

## Verdict

**PROMOTES.** BW10_GPTQ (BW8 + LOOP_AWARE_GPTQ=1) is the new crawler champion at 1.18292670 BPB.

Next: seed=300 confirmation, then update LEADER.md.
Next experiment: BW12 = BW11_5Flat (5F) + GPTQ — stack both improvements.
