# Speed Analysis: Strategic Summary

## The Core Insight

**76% of step time is overhead, not compute.** At 43.54ms/step, only ~10.5ms is actual
matrix multiplication on the H100. The rest is kernel launch overhead, memory operations,
and communication. This means:

1. **Speed optimizations have high ceiling** — could theoretically 2-3x throughput
2. **But most individual optimizations give only 1-10%** — the overhead is distributed
3. **Architecture choice matters more than code optimization** for total training quality

## Two Competing Strategies

### Strategy A: Recurrence + Speed Boost
- 7×2@672: 0.018 BPB better per 2K steps, but 2.45x slower
- With 25% speed boost: ~7,500 steps (vs 13,780 baseline)
- Total compute: 1.33x baseline (25% more FLOPs seen)
- Risk: shared weights may not scale as well as unique weights
- **Requires significant engineering effort** for speed optimizations

### Strategy B: Deeper Non-Recurrent (NO speed work needed)
- 12@416 or 13@400: 3-4 MORE layers, FASTER than baseline
- Gets 15,600+ steps (vs 13,780 baseline = +1,800 extra steps!)
- Total compute: 1.00x baseline (same FLOPs but more layers)
- All weights are unique (better diversity, proven to work)
- **Zero engineering effort** — just change hyperparameters
- Risk: narrower model may hurt per-layer quality

### Strategy C: Moderate Adjustments + Quick Speed Wins
- 10@480 or 11@464: 1-2 extra layers at similar speed
- Apply easy speed optimizations (reduce-overhead, val frequency, NS steps)
- Gets 14,000-15,000 steps with minimal code changes
- **Safest bet** — incremental improvement with low risk

## Recommended Testing Order

1. **Test 12@416 8h4kv at 2K steps** — zero-cost deeper model
   - If it beats baseline at 2K, it will likely beat at 12K+ steps too
   - `NUM_LAYERS=12 MODEL_DIM=416 NUM_HEADS=8 NUM_KV_HEADS=4`
   
2. **Test 11@464 8h4kv at 2K steps** — safest depth increase
   - `NUM_LAYERS=11 MODEL_DIM=464 NUM_HEADS=8 NUM_KV_HEADS=4`

3. **Test 13@400 4h4kv at 2K steps** — aggressive depth for speed
   - `NUM_LAYERS=13 MODEL_DIM=400 NUM_HEADS=4 NUM_KV_HEADS=4`

4. **If recurrence wins quality tests**, apply speed optimizations:
   - `mode='reduce-overhead'` in torch.compile
   - `VAL_LOSS_EVERY=5000`
   - `MUON_BACKEND_STEPS=3`
   - `static_graph=True` in DDP

## Key Numbers

| Config | ms/step | Steps/10min | Extra Layers | Params | Fits 16MB |
|--------|---------|-------------|-------------|--------|-----------|
| 9@512 (baseline) | 43.5 | 13,780 | — | 17.06M | ✓ |
| 11@464 | 43.7 | 13,728 | +2 | 17.08M | ✓ |
| 12@416 4h4kv | 38.4 | 15,655 | +3 | 17.06M | ✓ |
| 13@400 4h4kv | 38.4 | 15,630 | +4 | 17.07M | ✓ |
| 7×2@672 | ~107 | ~5,600 | +5 eff | 16.86M | ✓ |
| 7×2@672 + 25% boost | ~80 | ~7,500 | +5 eff | 16.86M | ✓ |

## Validation Frequency Savings (Often Overlooked!)

Current: VAL_LOSS_EVERY=1000 → 14 validations in 13,780 steps
Each validation takes ~2s → 28s total → ~640 steps lost!

With VAL_LOSS_EVERY=5000 → 3 validations → 6s → ~140 steps lost
**Net gain: ~500 extra training steps for FREE!**

This alone could improve BPB by ~0.002 from the extra training.
