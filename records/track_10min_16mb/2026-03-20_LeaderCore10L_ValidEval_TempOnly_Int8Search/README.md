This directory contains the leader-core merge candidate rooted in the official `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` training recipe, with validity-safe final eval and the stronger local int8 export plus temperature-only post-quant search.

Primary file:
- `train_gpt.py`

Saved full 8xH100 parity result:
- `runpod_tmp_base`: `final_int8_zlib_roundtrip_exact val_bpb: 1.20639536`
- `runpod_tmp_base`: `Total submission size int8+zlib: 15294320 bytes`
- `runpod_tmp_base`: `step_avg: 49.31ms`

Saved 1xH100 proxy screens on the same line:
- `screen_tmp_base`: `val_bpb: 1.7975`, `step_avg: 472.64ms`
- `screen_tmp_gradclip03`: `val_bpb: 1.7464`, `step_avg: 477.81ms`
- `screen_tmp_matrixlr006`: `val_bpb: 1.6751`, `step_avg: 471.55ms`
- `screen_tmp_muon099`: `val_bpb: 1.6582`, `step_avg: 416.73ms`
- `screen_tmp_warmdown800`: `val_bpb: 1.5984`, `step_avg: 523.52ms`
- `screen_tmp_warmdown800_matrixlr006`: `val_bpb: 1.4874`, `step_avg: 423.34ms`
- `screen_tmp_embedlr08`: `val_bpb: 1.8000`, `step_avg: 469.15ms`

Current proxy read:
- strongest single change: `warmdown800`
- strongest low-risk single change: `matrixlr006`
- strongest tested combination so far: `warmdown800 + matrixlr006`

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
- `matrixlr006`: `MATRIX_LR=0.06`
- `warmdown1800`: `WARMDOWN_ITERS=1800`
- `warmdown800`: `WARMDOWN_ITERS=800`
- `warmdown800_matrixlr006`: `WARMDOWN_ITERS=800`, `MATRIX_LR=0.06`
- `gradclip03`: `GRAD_CLIP_NORM=0.3`
- `muon099`: `MUON_MOMENTUM=0.99`
- `tokemb_int8`: `INT8_KEEP_TOK_EMB_FP16=0`

Data root modes:
- `workspace`: use `/workspace/parameter-golf/data/...`
- `tmp`: use `/tmp/parameter-golf-data/...` after local staging
