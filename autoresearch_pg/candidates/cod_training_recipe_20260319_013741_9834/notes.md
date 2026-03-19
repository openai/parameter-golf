# cod_training_recipe_20260319_013741_9834

1. Hypothesis

Keep the parent LR-floor recipe, but add a late warmdown export-projection phase: every few updates, snap only the large 2D tensors that will actually ship as int8 (`tok_emb.weight` plus the Muon-managed block matrices) back onto their exact int8-dequantized grid. This should train the late recipe around the lattice the model will be evaluated on, reducing the remaining roundtrip gap without changing bytes or disturbing the early fast-learning phase.

2. Expected upside

Lower `final_int8_zlib_roundtrip_exact val_bpb` by shrinking post-quant drift on the largest exported tensors while leaving the inherited early and mid-training behavior intact. Because the projections only begin in the last half of warmdown and only every 8 optimizer steps, the wallclock tax should stay modest relative to always-on fake quantization.

3. Expected risk

Periodic snapping can add late optimization noise and some CPU-side quantization overhead, which may reduce total completed steps under the 600s cap or slightly hurt raw validation if the trigger is too aggressive. If the remaining gap is dominated by float-kept control tensors instead of the large int8 tensors, the gain may be small.

4. Exact knobs changed

Added `QAT_INTERVAL=8` and `QAT_TRIGGER_LR_MUL=0.5` defaults in `train_gpt_mlx.py`. Added `qat_should_project()` plus `project_large_matrices_to_export_grid()`, reusing the exact existing int8 quantizer/dequantizer. After each optimizer step, when `base_lr_mul <= 0.5` and `(step + 1) % 8 == 0`, the script projects `tok_emb.weight` and `opt.matrix_keys` back through the export grid and logs both the config and total projection counts.

5. Promotion bar

`smoke_local`: no runtime breakage, logs show the QAT projection knobs and nonzero total projections when warmdown is reached, and submission bytes stay effectively unchanged. `proxy_1gpu_fast`: beat the parent on `final_int8_zlib_roundtrip_exact val_bpb`, or at minimum reduce the raw-vs-roundtrip gap without giving back too much raw validation.
