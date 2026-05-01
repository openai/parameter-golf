# Ablations

These are the 56 single-run ablation scripts that informed the submitted configuration. Each file is a self-contained training script tied to one row in the *Result summary* table of the parent `../README.md`. Filenames roughly map to the variant names in that table:

- `ssm_recall_sota_sp8192_fullmuon_final_compress.py` — the submitted file (also a copy at `../train_gpt.py`)
- `ssm_recall_sota_sp8192_fullmuon_final_compress_trustclip.py` — Run #3 (trust-clipped Muon variant) referenced in the README
- `ssm_recall_sota_sp8192_race_best*.py` — sparse / shared / fused FFN attempts
- `ssm_recall_sota_sp8192_j_tieddense*.py` — tied-FFN J family
- `ssm_recall_sota_sp8192_k35_*.py` — K35 narrow-FFN family (TaskMuon, AdamW, etc.)
- `ssm_recall_sota_sp8192_*ttt*.py` — TTT variants (LoRA, dynamic bias, score-first)
- `ssm_recall_sota_sp8192_hybrid_*.py` / `*_fusion*` / `*_sidecar.py` — attention-fusion attempts
- `ssm_recall_sota_sp8192_*_compactffn*.py` — compact / SVD-rank FFN
- `ssm_recall_sota_sp8192_h3_compare.py`, `*_long_s4d.py`, `*_smeargate.py` — alt SSM/architecture variants
- `ssm_recall_sota_sp8192_*_curriculum.py`, `*_rope*.py`, `*_tail_loss.py` — training-recipe variants
- `ssm_recall_sota_sp8192_nbit_quant.py`, `*_byteint6_lzma.py`, `*_hybrid_int5_lzma.py`, `*_bf16storage.py` — quantization/storage variants
- `ssm_param_golf_fixed.py`, `ssm_param_golf_repair.py`, `ssm_recall.py`, `ssm_ropebridge_taskmuon.py` — earlier exploratory baselines

These are kept here for full provenance — most are exploratory and not tuned. The submitted configuration is `../train_gpt.py`. See the parent README's *Result summary* and *Architecture/Optimizer/Compression learnings* sections for the conclusions drawn from these runs.
