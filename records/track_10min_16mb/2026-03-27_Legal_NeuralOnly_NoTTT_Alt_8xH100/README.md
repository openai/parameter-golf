# Legal Neural-Only No-TTT Alt (8xH100)

## Summary

Alternative compliance-focused hedge run with a larger neural configuration, while still avoiding the criticized cache-rescoring pattern.

Key properties:
- Neural model only during eval (`NGRAM_EVAL_ENABLED=0`, `NGRAM_TWO_PASS_ENABLED=0`, `NGRAM_FULL_RESCORE=0`)
- No test-time training (`TTT_ENABLED=0`)
- No tokenizer or dataset modifications
- Score-first causal evaluation flow from the base script

## Run Configuration

- Model preset: `frontier_candidate`
- Profile: `full_8gpu_600s`
- Seed: `1337`
- Overrides: `BIGRAM_VOCAB_SIZE=2048`, `MLP_MULT=3.2`
- Train cap: `MAX_WALLCLOCK_SECONDS=563`
- Eval cap guard: `EVAL_TIME_SOFT_CAP_SECONDS=570`
- Artifact soft cap guard: `SUBMISSION_SOFT_CAP_BYTES=15800000`
- Sliding eval disabled for bounded runtime: `SKIP_SLIDING_EVAL=1`

## Results

- `final_research_export_exact val_loss: 1.95453440`
- `final_research_export_exact val_bpb: 1.15758536`
- Pre-quant diagnostic `val_bpb: 1.1399`
- Train time: `563.076s`
- Roundtrip eval time: `44.296s`
- Total submission size: `14,921,440 bytes`
- Serialized model size: `14,713,448 bytes`

## Included Files

- `train_gpt.py`
- `train_seed1337.log`
- `submission.json`
- `README.md`
