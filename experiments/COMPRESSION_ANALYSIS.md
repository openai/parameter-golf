# Compression Analysis — ChatGPT Recommendations Assessment

## Our Problem
Artifact: ~18MB with torch.save+zstd. Need ≤16MB. Gap: ~2MB.
PR135 gets 15.16MB with same model on different hardware.
Our manual serialization (exp063 running) should help by ~2-3MB.

## Recommendation Assessment

### 1. Serializer reordering by payload type — EASIEST, DO FIRST
**Verdict: YES — implement immediately**
- Currently sorting by key name. Instead: group all .q int8 tensors first, then all .scale fp16 tensors, then all passthrough fp16 tensors.
- Similar dtype bytes together = better zstd compression.
- Zero quality impact, zero training impact. Pure compression win.
- Estimate: 0.5-1MB savings on top of manual serialization.

### 2. Outlier splitting — HIGH ROI but needs careful implementation
**Verdict: TRY AFTER serializer reordering**
- Keep top-K outlier weights in fp16, zero them in bulk, quantize residual.
- The residual becomes more regular → compresses better.
- Also IMPROVES quality because outliers are preserved at higher precision.
- Risk: adds complexity to serialize/deserialize. More tensors to manage.
- Could save ~0.5-1MB in compression + 0.001-0.002 BPB quality improvement.

### 3. Blockwise int6 scales — MODERATE ROI
**Verdict: MAYBE — lower priority than 1 and 2**
- Currently per-row scales. Block-32 or block-64 would use more scale bytes but better quantization.
- More scales = more bytes, may not help compression.
- Quality improvement is marginal at int6 (already pretty good).
- Would need to benchmark carefully.

### 4. Sensitivity-based mixed precision — ALREADY PARTIALLY DONE
**Verdict: We already do this (fp16 for tok_emb and last 2 K projections)**
- PR135/PR114 identified the sensitive tensors empirically.
- Could extend: measure per-tensor quant error and keep more in fp16 if budget allows.
- Low priority since we already cover the big ones.

### 5. Codebook/additive quantization — TOO INVASIVE
**Verdict: SKIP for now**
- Would require major code changes to encode/decode.
- We're at int6 which is already quite low. Codebook would help at 2-3 bits but we're at 6.
- Engineering complexity not justified given our artifact gap is ~2MB (solvable with simpler methods).

## Action Plan
1. ✅ Manual serialization (exp063 running — tests contiguous bytes + zstd)
2. **NEXT**: If manual serialization doesn't fully solve it, add dtype-grouped ordering
3. **THEN**: Outlier splitting for remaining gap
4. All of these can be stacked

## Note
The 2MB gap may vanish on Runpod (official eval platform) due to different PyTorch version.
Test on Runpod ASAP to check actual submission artifact size.
