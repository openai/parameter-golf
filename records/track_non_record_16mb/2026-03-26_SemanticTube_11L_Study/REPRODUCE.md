# Reproduction

The confirmatory runner is self-contained:
- it uses the `train_gpt.py` included in this folder
- it ensures the full `fineweb10B_sp1024` dataset is present (`80` training shards)
- it installs or reuses a Hopper-only FA3 wheel before training
- it trains in compiled mode and evaluates with the included trajectory-metric logging path

## Confirmatory Runs

Run the full confirmatory matrix:

```bash
SKIP_QUANT=0 \
  bash records/track_non_record_16mb/2026-03-26_SemanticTube_11L_Study/run_semantic_tube_public_matrix.sh
```

This reruns four matched comparisons with the real quantization/artifact path enabled:
- `T0`: `seq1024`, `lambda=0`
- `T4`: `seq1024`, `lambda=5e-4`
- `S2`: `seq2048`, `lambda=0`
- `S3`: `seq2048`, `lambda=5e-4`

These reruns are the authoritative public throughput/score path for the study. The discovery/family sweep remains useful as within-family evidence, but the public confirmatory logs should be used for the final absolute numbers reported in the README and PR.

The confirmatory runners copy their exact training logs into `public_logs/` inside this folder.
