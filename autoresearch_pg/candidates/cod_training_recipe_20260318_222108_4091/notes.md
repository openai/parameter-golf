# cod_training_recipe_20260318_222108_4091

1. Hypothesis

Keep the parent scalar-tail recipe, but add a much smaller nonzero LR floor to the tied embedding Adam group so the export-critical shared token/logit table can keep co-adapting with the float-kept control tensors after the large quantized block matrices have mostly frozen.

2. Expected upside

Reduce `final_int8_zlib_roundtrip_exact val_bpb` by narrowing the remaining pre-vs-post-quant gap. The tied embedding is both the input table and the output projection, so a light tail here should help the parent scalar calibration land on a logit surface that still matches the quantized artifact path.

3. Expected risk

If the tied embedding keeps moving too much after the block matrices decay, it could overfit token-frequency quirks or make the quantized embedding table less stable, hurting either raw validation or the post-quant gap.

4. Exact knobs changed

Added `TIED_EMBED_LR_FLOOR_RATIO` in `train_gpt_mlx.py`, defaulting to `0.05`.
Refactored the LR-floor logic into a shared helper so:
- matrix Muon LR still uses the existing base `lr_mul(...)`
- tied embedding Adam LR now uses `tied_embed_lr_mul = floor + (1 - floor) * lr_mul`
- scalar/control Adam LR keeps using its parent `scalar_lr_mul`

This intentionally keeps the inherited parent env overrides unchanged:
- `WARMUP_STEPS=64`
- `WARMDOWN_ITERS=256`
- `MATRIX_LR=0.05`
- `SCALAR_LR=0.04`
- `SCALAR_LR_FLOOR_RATIO=0.25`

5. Promotion bar

`smoke_local`: no runtime breakage, logs show both `embed_lr_floor_ratio` and `scalar_lr_floor_ratio`, and artifact bytes stay effectively unchanged.
`proxy_1gpu_fast`: beat the parent on `final_int8_zlib_roundtrip_exact val_bpb`, or at minimum trim the remaining quant gap without giving back more than a negligible amount of raw validation quality.
