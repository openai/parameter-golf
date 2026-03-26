# NovelStack Final Local MLX Stack

This branch contains a single local MLX stack focused on:

- TriShift causal input mixer
- explicit MLP hidden size (`MLP_HIDDEN_DIM=1408`)
- decoupled weight decay for Muon and Adam groups
- int8+zlib export with optional selective fp16 passthrough

## Run command

```bash
PYTHON=.venv/bin/python bash code/run_best_novelstack.sh
```

## Final stack defaults

- `RUN_ID=mlx_novel_trishift_final`
- `NUM_LAYERS=10`
- `MODEL_DIM=512`
- `MLP_MULT=2.75`
- `MLP_HIDDEN_DIM=1408`
- `INPUT_MIX_ENABLED=1`
- `HARMONIC_EMBED_INIT=0`
- `PHASE_RESID_MIX_INIT=0`
- `MUON_WEIGHT_DECAY=0.018`
- `EMBED_WEIGHT_DECAY=0.012`
- `SCALAR_WEIGHT_DECAY=0.0`

## Current observed local run metrics

- `val_loss: 5.7055`
- `val_bpb: 3.3786`
- `serialized_model_int8_zlib: 6,804,703 bytes`
- `train_time_ms: 23,781`
- `step_avg_ms: 1189.05`
