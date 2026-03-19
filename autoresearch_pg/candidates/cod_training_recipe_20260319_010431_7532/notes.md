# cod_training_recipe_20260319_010431_7532

1. Hypothesis

Keep the parent scalar-tail, tied-embedding-tail, and tiny Muon matrix floor, but add a staged warmdown-only int8 projection for the Muon block matrices. The idea is to periodically snap those large export-quantized weights onto the exact per-row int8 roundtrip grid during the late low-LR phase so the remaining tiny matrix updates learn to live on the deployed lattice instead of only the float path.

2. Expected upside

Reduce `final_int8_zlib_roundtrip_exact val_bpb` by trimming the remaining post-quant gap without increasing bytes. This specifically targets the family-best failure mode where the run is already strong pre-quant, but the large block matrices still move in float space right up until export.

3. Expected risk

If the warmdown projection is too frequent, it can inject enough discretization noise to give back some of the raw-loss gain from the parent recipe or make the short final phase too stiff to finish co-adapting cleanly.

4. Exact knobs changed

- Added `MATRIX_QAT_INTERVAL` in `train_gpt_mlx.py`, defaulting to `8`.
- Added a warmdown-only `maybe_project_matrix_int8_roundtrip(...)` helper that periodically quantizes and dequantizes only the Muon-managed block matrices using the same per-row int8 path as final export.
- Logged `matrix_qat_interval`, the first activation step, and the total number of projected tensors so smoke and promotion runs can confirm the behavior.
- Kept the inherited env overrides and parent LR-floor behavior unchanged:
  `MATRIX_LR=0.05`, `SCALAR_LR=0.04`, `WARMDOWN_ITERS=256`, `WARMUP_STEPS=64`, `TIED_EMBED_LR_FLOOR_RATIO=0.05`, `MATRIX_LR_FLOOR_RATIO=0.01`, `SCALAR_LR_FLOOR_RATIO=0.25`.

5. Promotion bar

`smoke_local`: no runtime breakage, logs show `matrix_qat_interval`, `matrix_qat:active` during warmdown, and bytes stay effectively unchanged.
`proxy_1gpu_fast`: beat the parent on `final_int8_zlib_roundtrip_exact val_bpb`, or at minimum reduce the quant gap with only a negligible pre-quant regression.
