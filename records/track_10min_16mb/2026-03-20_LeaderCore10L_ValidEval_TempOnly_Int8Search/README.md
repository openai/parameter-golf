This directory contains the leader-core merge candidate rooted in the official `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` training recipe, with validity-safe final eval and the stronger local int8 export plus temperature-only post-quant search.

Primary file:
- `train_gpt.py`

First ablation batch:
- `base`
- `embedlr08`
- `matrixlr005`
- `warmdown1800`
- `tokemb_int8`

RunPod launcher:
```bash
bash /workspace/parameter-golf/launch_leadercore_ablation_runpod.sh base
```

For timing-parity runs, first stage data/tokenizer to local disk on the pod:
```bash
bash /workspace/parameter-golf/setup_local_parity_data_runpod.sh
DATA_ROOT_MODE=tmp bash /workspace/parameter-golf/launch_leadercore_ablation_runpod.sh base
```

Variant names map to these env changes:
- `base`: no overrides beyond the merged defaults
- `embedlr08`: `TIED_EMBED_LR=0.08`
- `matrixlr005`: `MATRIX_LR=0.05`
- `warmdown1800`: `WARMDOWN_ITERS=1800`
- `tokemb_int8`: `INT8_KEEP_TOK_EMB_FP16=0`

Data root modes:
- `workspace`: use `/workspace/parameter-golf/data/...`
- `tmp`: use `/tmp/parameter-golf-data/...` after local staging
