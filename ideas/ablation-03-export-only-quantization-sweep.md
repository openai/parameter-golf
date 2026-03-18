# Ablation 3. Export-Only Quantization Sweep

## Goal

Measure whether the main fast win is in the exporter rather than in the model.

## Why This Is One of the First Ablations

The recorded pre-quant to post-roundtrip gap is large enough to justify focused export experiments. This is especially important because the challenge score is based on the artifact, not the high-precision checkpoint.

## Suggested Sweep

On one good checkpoint, vary only export behavior:

- `INT8_CLIP_PERCENTILE`
- outlier row rescue budget in fp16
- `INT8_KEEP_FLOAT_MAX_NUMEL`

Keep total artifact under `16,000,000` bytes.

## Decision Rule

If export-only changes recover a nontrivial amount of final exact roundtrip `val_bpb`, then smarter quantization should rank ahead of most architecture work.

## Recommendation

Run first
