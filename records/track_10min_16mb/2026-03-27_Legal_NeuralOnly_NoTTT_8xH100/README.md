# Legal Neural-Only No-TTT (8xH100)

## Summary

This run is a compliance-focused hedge submission that avoids the criticized target-token blending pattern.

Key properties:
- Neural model only during eval (`NGRAM_EVAL_ENABLED=0`, `NGRAM_TWO_PASS_ENABLED=0`, `NGRAM_FULL_RESCORE=0`)
- No test-time training (`TTT_ENABLED=0`)
- No tokenizer or dataset modifications
- Causal evaluation with score-first ordering preserved by the base script

## Run Configuration

- Model preset: `frontier_lean`
- Profile: `full_8gpu_600s`
- Seed: `1337`
- Train cap: `MAX_WALLCLOCK_SECONDS=563`
- Eval cap guard: `EVAL_TIME_SOFT_CAP_SECONDS=570`
- Artifact soft cap guard: `SUBMISSION_SOFT_CAP_BYTES=15800000`
- Sliding eval disabled for bounded runtime: `SKIP_SLIDING_EVAL=1`

## Results

- `final_research_export_exact val_loss: 1.95961204`
- `final_research_export_exact val_bpb: 1.16059263`
- Pre-quant diagnostic `val_bpb: 1.1428`
- Train time: `563.039s`
- Roundtrip eval time: `6.492s`
- Total submission size: `13,446,760 bytes`
- Serialized model size: `13,238,768 bytes`

## Included Files

- `train_gpt.py`
- `train_seed1337.log`
- `submission.json`
- `README.md`
