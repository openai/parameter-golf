# cod_training_recipe_20260318_235033_8561

1. Hypothesis

Keep the parent scalar-tail plus tied-embedding-tail recipe, but add a much smaller nonzero LR floor to the Muon matrix group so the large quantized block weights can keep making tiny late co-adaptation moves instead of fully freezing ahead of the float-kept control tensors and tied embedding table.

2. Expected upside

Reduce `final_int8_zlib_roundtrip_exact val_bpb` by trimming the remaining post-quant gap. The current family already improved raw validation materially, but the roundtrip gap is still around `0.0047` BPB, which suggests the quantized block matrices may be lagging the late calibration happening in the float-kept controls and shared embedding/logit table.

3. Expected risk

If the matrix tail is even slightly too large, it could give back the fast-decay stability that made this family work, causing late noise in the biggest quantized tensors and hurting either raw validation or the roundtrip score.

4. Exact knobs changed

Added `MATRIX_LR_FLOOR_RATIO` in `train_gpt_mlx.py`, defaulting to `0.01`.
Wired Muon to use `matrix_lr_mul = floor + (1 - floor) * lr_mul`, while keeping the inherited parent behavior unchanged for:
- tied embedding Adam via `tied_embed_lr_mul`
- scalar/control Adam via `scalar_lr_mul`

This intentionally composes with the inherited env overrides rather than replacing them:
- `WARMUP_STEPS=64`
- `WARMDOWN_ITERS=256`
- `MATRIX_LR=0.05`
- `SCALAR_LR=0.04`

The logging line now prints `matrix_lr_floor_ratio` alongside the existing optimizer-group settings.

5. Promotion bar

`smoke_local`: no runtime breakage, logs show `matrix_lr_floor_ratio`, and submission bytes stay effectively unchanged.
`proxy_1gpu_fast`: beat the parent on `final_int8_zlib_roundtrip_exact val_bpb`, or at minimum reduce the raw-vs-roundtrip gap without giving back most of the parent's raw-validation gain.
