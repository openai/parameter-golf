Hypothesis: keep the parent LR-floor recipe, but during warmdown periodically snap only the Muon-managed block matrices onto the exact int8 export roundtrip grid so the last low-LR updates adapt to the deployed lattice instead of pure float space.

Changed files: [train_gpt_mlx.py](/Users/calmdentist/Desktop/parameter-golf/autoresearch_pg/candidates/cod_training_recipe_20260319_010431_7532/train_gpt_mlx.py#L90) and [notes.md](/Users/calmdentist/Desktop/parameter-golf/autoresearch_pg/candidates/cod_training_recipe_20260319_010431_7532/notes.md#L1). In `train_gpt_mlx.py` I added `MATRIX_QAT_INTERVAL`, a shared dequantize helper, and a warmdown-only periodic matrix int8 roundtrip projection hooked into the train step and logs.

Expected upside is a smaller post-quant gap, so better `final_int8_zlib_roundtrip_exact val_bpb` with essentially unchanged bytes. Main risk is that projecting too often makes the short late phase too stiff and gives back some pre-quant/raw validation quality.

Verification: `python3 -m py_compile train_gpt_mlx.py` passed. I did not run training.