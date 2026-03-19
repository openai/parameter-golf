# cod_training_recipe_20260319_012138_9154

1. Hypothesis

Keep the parent matrix-LR floor recipe, but add a staged late-phase int8 roundtrip snap on the large Muon-managed block matrices so those weights spend the final warmdown adapting to the exact per-row export distortion instead of only float-space updates.

2. Expected upside

Reduce `final_int8_zlib_roundtrip_exact val_bpb` by trimming the remaining quantization gap without changing artifact bytes. This specifically targets the currently meaningful post-quant loss by making the largest exported block matrices co-adapt under their deployed int8 approximation during the last part of training.

3. Expected risk

If the periodic snaps are too abrupt or start too early, they can inject late optimization noise, give back some of the parent's raw validation gain, or overspecialize the block matrices to the fake-quant perturbation while the tied embedding and float-kept control tensors continue moving more smoothly.

4. Exact knobs changed

Added three default-on knobs in `train_gpt_mlx.py`:
- `MATRIX_INT8_SNAP_START_LR_MUL=0.5`
- `MATRIX_INT8_SNAP_INTERVAL=8`
- `MATRIX_INT8_SNAP_MIN_NUMEL=131072`

Wired `SplitOptimizers.step(...)` to periodically roundtrip only the large `blocks.*` 2D Muon matrices through the existing `quantize_float_array` / dequantize path during late warmdown, while leaving:
- the inherited parent `MATRIX_LR_FLOOR_RATIO` behavior intact
- tied embedding Adam behavior unchanged
- scalar/control Adam behavior unchanged
- final export format and submission bytes unchanged

The run log now prints the staged int8-snap settings plus the number of targeted tensors.

5. Promotion bar

`smoke_local`: no runtime breakage, logs show the staged int8-snap knobs, and bytes remain effectively unchanged.
`proxy_1gpu_fast`: beat the parent on `final_int8_zlib_roundtrip_exact val_bpb`, or at minimum shrink the quant gap relative to the parent without giving back most of the raw-validation gain.
