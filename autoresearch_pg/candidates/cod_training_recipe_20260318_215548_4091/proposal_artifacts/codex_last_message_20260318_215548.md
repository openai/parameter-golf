Hypothesis: keep the inherited fast-decay schedule for quantized matrices and tied embeddings, but give the scalar/control Adam group a small nonzero LR tail (`SCALAR_LR_FLOOR_RATIO=0.25`) so late training can keep calibrating the float-kept control tensors and narrow the pre-vs-post-quant gap.

Changed files:
- [train_gpt_mlx.py](/Users/calmdentist/Desktop/parameter-golf/autoresearch_pg/candidates/cod_training_recipe_20260318_215548_4091/train_gpt_mlx.py#L89): added `scalar_lr_floor_ratio`, a `scalar_lr_mul(...)` helper, and separate optimizer LR wiring so matrices/embeddings follow the base fast-decay schedule while scalar/control params keep a floored tail.
- [notes.md](/Users/calmdentist/Desktop/parameter-golf/autoresearch_pg/candidates/cod_training_recipe_20260318_215548_4091/notes.md#L3): recorded the hypothesis, expected upside, risk, exact knobs, and promotion bar.

Expected upside is better `final_int8_zlib_roundtrip_exact val_bpb` with effectively unchanged artifact bytes, since the extra adaptation is confined to the small float-kept control group. Main risk is over-tuning those scalars after the large matrices have mostly frozen, which could hurt either raw val or roundtrip val.

Verification: `python3 -m py_compile train_gpt_mlx.py` passed. I did not run training.