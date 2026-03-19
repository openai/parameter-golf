# cod_training_recipe_20260318_215548_4091

1. Hypothesis

Keep the parent fast-decay recipe for quantized matrices and tied embeddings, but give the scalar/control Adam group a nonzero tail LR floor so late training can keep calibrating the float-kept control tensors (`attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, `skip_weights`) after the heavy quantized weights have mostly frozen.

2. Expected upside

Reduce the pre-vs-post-quant gap by letting the small float-kept control tensors absorb late calibration error without reintroducing large matrix updates. This should be especially relevant under the inherited short 600s-style warmdown where matrix updates shut off early.

3. Expected risk

If the scalar group keeps moving too long, it may overfit the tail or destabilize layer balance after the matrices stop adapting, which could hurt both raw val and roundtrip val despite the lower-risk parameter count.

4. Exact knobs changed

Added `SCALAR_LR_FLOOR_RATIO` in `train_gpt_mlx.py`, defaulting to `0.25`.
Changed the optimizer step wiring so:
- matrix Muon LR uses the existing `lr_mul(...)`
- tied embedding Adam LR uses the existing `lr_mul(...)`
- scalar/control Adam LR uses `scalar_lr_mul = floor + (1 - floor) * lr_mul`

This is intended to run on top of the inherited parent env overrides in `meta.json`:
- `WARMUP_STEPS=64`
- `WARMDOWN_ITERS=256`
- `MATRIX_LR=0.05`
- `SCALAR_LR=0.04`

5. Promotion bar

`smoke_local`: no runtime breakage, logs show `scalar_lr_floor_ratio`, and artifact bytes stay effectively unchanged.
`proxy_1gpu_fast`: beat the parent on `final_int8_zlib_roundtrip_exact val_bpb`, or at minimum narrow the train-vs-roundtrip gap without giving back more than a negligible amount of raw validation quality.
