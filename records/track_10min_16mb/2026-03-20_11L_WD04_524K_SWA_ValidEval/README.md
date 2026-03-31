This directory contains a fresh `11L` frontier candidate built from the local validity-safe lane, updated to reflect the strongest portable March 20 frontier themes:

- `11` layers
- `MLP_MULT=3`
- `TRAIN_BATCH_TOKENS=524288`
- `MUON_WD=0.04`
- `ADAM_WD=0.04`
- late SWA
- non-overlapping validity-safe final eval
- local int8 export with temperature-only post-quant search

Primary file:
- `train_gpt.py`

Primary launcher:
- `/workspace/parameter-golf/launch_11l_frontier_runpod.sh`

Base RunPod command:
```bash
bash /workspace/parameter-golf/launch_11l_frontier_runpod.sh base
```

Timing-parity RunPod command:
```bash
bash /workspace/parameter-golf/setup_local_parity_data_runpod.sh
DATA_ROOT_MODE=tmp bash /workspace/parameter-golf/launch_11l_frontier_runpod.sh base
```

Variants currently encoded:
- `base`: default `11L + WD0.04 + SWA`
- `wd038`: lower both Muon and AdamW decay to `0.038`
- `no_swa`: disable SWA to isolate its contribution
- `matrixlr002`: lower matrix/scalar LR to `0.02`
- `tokemb_int8`: export token embedding as int8 instead of fp16 passthrough

This branch is meant to test the portable training-side frontier, not to chase a sliding-window-only headline score.
