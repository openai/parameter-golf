Hypothesis: keep the parent scalar-tail recipe, and add a smaller LR tail for the tied embedding group so the shared token/logit table can keep co-adapting with the float-kept control tensors after the large quantized block matrices have mostly decayed.

Changed files:
- [train_gpt_mlx.py](/Users/calmdentist/Desktop/parameter-golf/autoresearch_pg/candidates/cod_training_recipe_20260318_222108_4091/train_gpt_mlx.py#L87): added `TIED_EMBED_LR_FLOOR_RATIO` with a shared floored-LR helper, logged `embed_lr_floor_ratio`, and wired the tied embedding Adam group to use its own floored multiplier.
- [notes.md](/Users/calmdentist/Desktop/parameter-golf/autoresearch_pg/candidates/cod_training_recipe_20260318_222108_4091/notes.md#L1): filled in the candidate hypothesis, upside, risk, exact knobs, and promotion bar.

Expected upside is a smaller pre-vs-post-quant gap, which should help `final_int8_zlib_roundtrip_exact val_bpb` with effectively unchanged bytes. Main risk is that late embedding drift makes the quantized tied table less stable and hurts either raw val or roundtrip val.

Verification: `python3 -m py_compile train_gpt_mlx.py` passed. I did not run training.