You are preparing one new candidate for the OpenAI Parameter Golf challenge.

Make one coherent proposal in this candidate-local workspace.

Mission:
- minimize `final_int8_zlib_roundtrip_exact val_bpb`
- keep `bytes_total <= 16_000_000`
- keep experiments reproducible and promotable to the official `records/` format

Read first:
- `train_gpt_mlx.py`
- `notes.md`
- `../../program.md`
- `proposal_artifacts/codex_context_latest.json` if you need deeper history or raw context

Current family:
- family: training_recipe
- proposal_mode: brief
- tier: smoke_mlx_local
- parent_candidate_id: cod_training_recipe_20260318_235033_8561
- inherited_env_overrides: {"MATRIX_LR": 0.05, "SCALAR_LR": 0.04, "WARMDOWN_ITERS": 256, "WARMUP_STEPS": 64}
- family_progress: completed=9 valid=9 invalid=0 wins=7 best=2.84089884 last=3.04063857

Family focus:
- learning-rate schedule
- optimizer-group balance
- sample-efficiency under a short wallclock

Curated starter ideas:
- `mtp_aux_heads_train_only` [near_term | risk=medium]: Train-only multi-token heads. Add 2 auxiliary future-token heads on top of the shared trunk during training, then drop them before export so bytes stay nearly unchanged.
- `staged_qat_large_matrices` [near_term | risk=medium]: Staged fake-quant final phase. During the last 10-15% of training, periodically fake-quantize only the large 2D block matrices while keeping small control tensors in higher precision.

Scoreboard:
- same_tier_best: cod_training_recipe_20260318_235033_8561 tier=smoke_mlx_local score=2.840899 bytes=6.48MB valid=true
- family_best: cod_training_recipe_20260318_235033_8561 tier=smoke_mlx_local score=2.840899 gap=0.005699 bytes=6.48MB op=codex_proposal valid=true
- same_tier_family_best: cod_training_recipe_20260318_235033_8561 tier=smoke_mlx_local score=2.840899 gap=0.005699 bytes=6.48MB op=codex_proposal valid=true
- low_quant_gap_anchor: cod_training_recipe_20260318_215548_4091 tier=smoke_mlx_local score=2.889461 gap=0.004661 bytes=6.34MB op=codex_proposal valid=true
- byte_headroom_anchor: port_compression_20260318_210204_9293 tier=smoke_mlx_local score=3.237472 gap=0.029272 bytes=5.16MB op=logit_softcap_sweep valid=true
- cross_tier_global_best: first_candidate tier=smoke_local score=1.224366 bytes=15.86MB valid=true (informational only; do not rank it above same-tier evidence)

Recent local evidence:
- cod_training_recipe_20260319_012138_9154 tier=smoke_mlx_local score=3.040639 gap=0.006239 bytes=5.45MB op=codex_proposal valid=true
- cod_training_recipe_20260319_010431_7532 tier=smoke_mlx_local score=3.046383 gap=0.005883 bytes=5.45MB op=codex_proposal valid=true
- cod_training_recipe_20260318_235033_8561 tier=smoke_mlx_local score=2.840899 gap=0.005699 bytes=6.48MB op=codex_proposal valid=true
- cod_training_recipe_20260318_222108_4091 tier=smoke_mlx_local score=2.858634 gap=0.004734 bytes=6.38MB op=codex_proposal valid=true

Recent failure evidence:
- none yet

Recent operator pattern:
- codex_proposal x5, recipe_fast_decay x1

Observations:
- Current base already inherits env overrides {"MATRIX_LR": 0.05, "SCALAR_LR": 0.04, "WARMDOWN_ITERS": 256, "WARMUP_STEPS": 64}. Build incrementally unless you mean to replace that behavior in code.
- training_recipe family best still has a material quant gap (0.005699 BPB). Export-aware changes remain valuable.
- Use same-tier results as the main local compass. Current tier best is cod_training_recipe_20260318_235033_8561 at proxy_score=2.84089884.

Parent context:
- parent_best_run: cod_training_recipe_20260318_235033_8561 tier=smoke_mlx_local score=2.840899 gap=0.005699 bytes=6.48MB op=codex_proposal valid=true
- parent_hypothesis: Keep the parent scalar-tail plus tied-embedding-tail recipe, but add a much smaller nonzero LR floor to the Muon matrix group so the large quantized block weights can keep making tiny late co-adaptation moves instead of fully freezing ahead of the float-kept control tensors an...
- parent_knobs: Added `MATRIX_LR_FLOOR_RATIO` in `train_gpt_mlx.py`, defaulting to `0.01`. Wired Muon to use `matrix_lr_mul = floor + (1 - floor) * lr_mul`, while keeping the inherited parent behavior unchanged for: - tied embedding Adam via `tied_embed_lr_mul` - scalar/control Adam via `scal...
- parent_changed_symbols: matrix_lr_floor_ratio, MATRIX_LR_FLOOR_RATIO, matrix_lr_mul
- parent_last_proposal: Hypothesis: keep the parent scalar and tied-embedding LR tails, and add a much smaller Muon matrix LR floor so the large quantized block weights can still make tiny late co-adaptation updates instead of freezing before the float-kept controls and shared embedding/logit table....


Workspace rules:
- Edit only the candidate-local files listed below.
- Keep the change small enough that it is attributable.
- Do not launch training or any long-running experiments.
- Preserve runnability of `train_gpt_mlx.py`.
- If you change the hypothesis materially, update `notes.md` sections 1-4.
- Prefer same-tier signals over cross-tier scoreboard noise when they disagree.
- Keep the improvement compatible with the inherited env overrides unless you have a clear reason to change behavior in code.
- Prefer one targeted change that composes with the parent over a broad rewrite.

Allowed files:
- train_gpt.py
- train_gpt_mlx.py
- notes.md

Primary task:
- Inspect `train_gpt_mlx.py`.
- Implement one promising improvement for the training_recipe family.
- Prefer a small but real code change over commentary.
- Do not touch files outside this candidate workspace.
- Use the context above to avoid duplicating already-tried weak ideas.

Final response:
- Briefly state the hypothesis.
- Mention the exact files you changed.
- Mention expected upside and main risk.
