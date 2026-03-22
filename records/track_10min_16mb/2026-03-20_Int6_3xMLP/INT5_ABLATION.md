# Ablation: Int5 Quantization (QUANT_BITS=5, MLP_HIDDEN=1920)

**Date:** 2026-03-20
**Script:** `records/track_10min_16mb/2026-03-20_Int6_3xMLP/train_gpt.py`
**Config:** `QUANT_BITS=5 MLP_HIDDEN=1920 SEED=1337`
**Verdict:** ❌ Negative — catastrophic quantization degradation

---

## Results

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb (step 10200, wallclock stop) | **1.1885** |
| Post-quant val_bpb (int5 roundtrip) | **1.5458** |
| Quantization gap | **+0.357 bpb** |
| Step avg | 58.9ms |
| Steps completed | 10,200 (600s cap) |

### Comparison vs Int6

| | Pre-quant val_bpb | Post-quant val_bpb | Quant gap |
|---|---|---|---|
| **Int6** (our submission) | 1.1949 | **1.1708** | +0.024 |
| **Int5** (this ablation) | 1.1885 | 1.5458 | +0.357 |

Int5's quantization gap is **15× larger** than int6's.

---

## Why Int5 Fails

Int5 per-row quantization maps weights to the range `[-15, 15]` — only **31 levels**.
Int6 uses `[-31, 31]` — **63 levels**.
Int8 uses `[-127, 127]` — **255 levels**.

Each step down halves the number of representable values. The jump from int6 to int5 is the same relative reduction as int8 to int6, but int6 was already at the edge of viability. With only 31 levels, per-row quantization error becomes large enough to destroy the model's language modeling ability entirely.

The pre-quant score (1.1885) is actually slightly **better** than our int6 run's pre-quant (1.1949) — int5 frees enough artifact budget to fit MLP_HIDDEN=1920 (vs 1536 for int6), which gives more parameters and better raw training. But the quantization destroys all of that gain and then some.

## Side note: Triton QAT warnings

The run produced warnings:
```
tl.where with a non-boolean condition is deprecated... Got int8
```

These come from the QAT (quantization-aware training) triton kernel, which still uses int8 internally regardless of `QUANT_BITS`. QAT and the export quantization are independent: QAT simulates int8 during training, while `QUANT_BITS=5` only affects the post-training export serialization. This means int5 QAT (simulating 5-bit during training) was never actually tested — that would require modifying the QAT kernel itself.

---

## Conclusion

**Int5 is not viable with current post-training quantization.** The quantization gap (+0.357) far outweighs any capacity gain from the extra artifact headroom.

A potential future path would be **Int5 QAT** — actually simulating 5-bit quantization during forward passes so the model learns to be robust to int5 rounding. This would require modifying the QAT triton kernel to use 5-bit fake quantization instead of 8-bit. Given that even int8 QAT was a negative finding (PR #145), this is unlikely to be worth pursuing.

**Next experiment:** 11L + FA2 + WD=0.04 (`11l-fa3-wd04` branch)
