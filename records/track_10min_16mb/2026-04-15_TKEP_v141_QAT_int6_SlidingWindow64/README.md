# TKEP v1.4.1 + QAT int6 + Sliding Window Eval (stride=64)

## Result
- **val_bpb: 2.25851226** (RunPod RTX 4090)
- Filter delta vs own baseline: +0.0663 (strongest signal in experiment series)

## Approach
Three load-bearing components that stack:

1. **TKEP quality filtering** — scores 50k FineWeb docs with a causal density
   scorer (v1.4.1). Selects high-signal documents where concepts cause other
   concepts. Adaptive threshold [0.31 → 0.305 → 0.30], random order, 90% fill.

2. **QAT int6** — fake-quantize all Linear layer weights to int6 range [-31,31]
   via forward pre-hooks during training. Straight-through gradient.

3. **Sliding window eval** — EVAL_STRIDE=64, each scored token sees 960+ tokens
   of context vs ~512 standard. Zero training cost, purely better measurement.

## Config
- Vocab: 8192 (SentencePiece)
- Pool: 50,000 FineWeb docs (pre-scored cache)
- Steps: 1500, seed: 42
- Hardware: RunPod RTX 4090
