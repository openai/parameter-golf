# Parameter Golf Research Program — Storage Export Lane

## Objective
Win on the final exported artifact by exploring mixed-precision and export-path changes that reduce bytes or reduce export damage.

## Primary Principle
Some matrices are much more sensitive than others. Do not assume one export format should be used everywhere.

## What We Know
- Strong model families are already close pre-quant and often lose at export time.
- Public leaders are getting value from FP16 tied embeddings and other export-aware decisions.
- Small byte increases can be worthwhile if they recover more post-export BPB than they cost.

## Priority Order
1. FP16 or mixed-precision export for the most sensitive matrices
2. Matrix-specific quantization or clipping choices
3. Byte-neutral or byte-positive export changes with clear BPB upside
4. Compression-friendly packing/layout improvements

## Preferred Directions
- Keep tied embeddings or LM-head weights in FP16 while quantizing less sensitive matrices
- Use matrix-specific or row-family-specific export precision rules
- Reduce scale/metadata overhead where possible
- Try export-path logic that preserves embedding/logit quality
- Consider zlib-friendly packing only if it is simple and measurable

## Avoid
- Broad training-loop changes unless they are explicitly export-aware
- Large architecture changes
- Ideas that only improve pre-quant loss
- Complicated compression schemes with high code overhead

## Guidance
- Make one conceptual change at a time
- Judge success by post-export `val_bpb` and artifact bytes together
- It is acceptable to spend some byte budget if the post-export BPB gain is clearly worth it
- Focus especially on the embedding / output projection path
