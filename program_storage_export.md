# Parameter Golf Research Program — Storage Export Lane

## Objective
Win on the final exported artifact by exploring mixed-precision and export-path changes that reduce bytes or reduce export damage.

## Primary Principle
Some matrices are much more sensitive than others. Do not assume one export format should be used everywhere.

## What We Know
- Strong model families are already close pre-quant and often lose at export time.
- Public leaders are getting value from FP16 tied embeddings and other export-aware decisions.
- Small byte increases can be worthwhile if they recover more post-export BPB than they cost.
- Recent competitive PRs repeatedly converge on:
  - int6 for most weights
  - fp16 tied embeddings
  - selective fp16 passthrough for especially sensitive matrices
  - QAT / fake-quant training as a major quantization-gap reducer

## Priority Order
1. FP16 or mixed-precision export for the most sensitive matrices
2. Selective precision for embeddings / late sensitive projections
3. Int6-style export directions or approximations toward them
4. Matrix-specific quantization or clipping choices
5. Byte-neutral or byte-positive export changes with clear BPB upside
6. Compression-friendly packing/layout improvements

## Preferred Directions
- Keep tied embeddings or LM-head weights in FP16 while quantizing less sensitive matrices
- Try selective fp16 passthrough for especially sensitive projections instead of all-or-nothing precision
- Use matrix-specific or row-family-specific export precision rules
- Reduce scale/metadata overhead where possible
- Try export-path logic that preserves embedding/logit quality
- Prefer same-day implementable versions of int6/mixed-precision and QAT-adjacent ideas over exotic codecs
- Consider zlib-friendly packing only if it is simple and measurable

## Avoid
- Broad training-loop changes unless they are explicitly export-aware
- Large architecture changes
- Ideas that only improve pre-quant loss
- Complicated compression schemes with high code overhead
- High-concept proposals without a concrete file-level change
- Anything that requires inventing a new codec tonight
- Multi-part changes that touch training, export, and eval simultaneously

## Guidance
- Make one conceptual change at a time
- Judge success by post-export `val_bpb` and artifact bytes together
- It is acceptable to spend some byte budget if the post-export BPB gain is clearly worth it
- Focus especially on the embedding / output projection path
- Prefer simple selective-precision edits over abstract compression plans
- If Claude cannot clearly name the exact tensors or export path being changed, the proposal is too vague
