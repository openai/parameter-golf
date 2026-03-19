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
- family: architecture
- proposal_mode: deep
- tier: smoke_mlx_local
- parent_candidate_id: port_architecture_20260319_023023_5357
- inherited_env_overrides: {"MLP_MULT": 1, "MODEL_DIM": 576, "NUM_HEADS": 9, "NUM_KV_HEADS": 3, "NUM_LAYERS": 6}
- family_progress: completed=7 valid=7 invalid=0 wins=4 best=3.24020662 last=3.24034187

Family focus:
- parameter efficiency
- depth/width/head tradeoffs
- challenge-native recurrence or sharing ideas

Curated starter ideas:
- `shared_depth_recurrence_3x_unroll` [medium_term | risk=high]: 3-block shared-depth recurrence. Replace many untied layers with 2-3 shared blocks unrolled multiple times, using step embeddings and per-step gates or norms.
- `factorized_embeddings_sp4096` [medium_term | risk=medium]: Factorized tied embeddings with larger vocab. Factor the token embedding/output map so the model can afford a 2k-4k tokenizer without paying a full vocab x model_dim byte tax.
- `mtp_aux_heads_train_only` [near_term | risk=medium]: Train-only multi-token heads. Add 2 auxiliary future-token heads on top of the shared trunk during training, then drop them before export so bytes stay nearly unchanged.

Scoreboard:
- same_tier_best: port_training_recipe_20260319_025140_4806 tier=smoke_mlx_local score=2.808115 bytes=6.37MB valid=true
- family_best: port_architecture_20260319_023023_5357 tier=smoke_mlx_local score=3.240207 gap=0.024907 bytes=2.95MB op=depth_sweep valid=true
- same_tier_family_best: port_architecture_20260319_023023_5357 tier=smoke_mlx_local score=3.240207 gap=0.024907 bytes=2.95MB op=depth_sweep valid=true
- low_quant_gap_anchor: port_training_recipe_20260319_022239_9771 tier=smoke_mlx_local score=2.917908 gap=0.002908 bytes=6.51MB op=embed_lr_sweep valid=true
- byte_headroom_anchor: port_architecture_20260319_020016_7076 tier=smoke_mlx_local score=3.246039 gap=0.024239 bytes=2.83MB op=mlp_sweep valid=true
- cross_tier_global_best: first_candidate tier=smoke_local score=1.224366 bytes=15.86MB valid=true (informational only; do not rank it above same-tier evidence)

Recent local evidence:
- cod_architecture_20260319_024400_5344 tier=smoke_mlx_local score=3.240342 gap=0.024942 bytes=2.96MB op=codex_proposal valid=true
- port_architecture_20260319_023023_5357 tier=smoke_mlx_local score=3.240207 gap=0.024907 bytes=2.95MB op=depth_sweep valid=true
- port_architecture_20260319_022908_5031 tier=smoke_mlx_local score=3.241801 gap=0.034401 bytes=6.36MB op=mlp_sweep valid=true
- port_architecture_20260319_022832_5187 tier=smoke_mlx_local score=3.240459 gap=0.029559 bytes=4.73MB op=arch_slim_10x576_mlp1 tpl=arch_slim_10x576_mlp1 valid=true

Recent failure evidence:
- none yet

Recent operator pattern:
- codex_proposal x2, mlp_sweep x2, arch_slim_10x576_mlp1 x1

Observations:
- Current base already inherits env overrides {"MLP_MULT": 1, "MODEL_DIM": 576, "NUM_HEADS": 9, "NUM_KV_HEADS": 3, "NUM_LAYERS": 6}. Build incrementally unless you mean to replace that behavior in code.
- architecture family best still has a material quant gap (0.024907 BPB). Export-aware changes remain valuable.
- Use same-tier results as the main local compass. Current tier best is port_training_recipe_20260319_025140_4806 at proxy_score=2.80811527.
- Best low-quant-gap candidate currently comes from training_recipe. Borrow export-stable habits if relevant.

Parent context:
- parent_best_run: port_architecture_20260319_023023_5357 tier=smoke_mlx_local score=3.240207 gap=0.024907 bytes=2.95MB op=depth_sweep valid=true
- parent_hypothesis: Scheduler cycle 29: action=mutate family=architecture, operator=depth_sweep, parent=port_architecture_20260319_022832_5187, env_overrides={'MLP_MULT': 1, 'MODEL_DIM': 576, 'NUM_HEADS': 9, 'NUM_KV_HEADS': 3, 'NUM_LAYERS': 6}
- parent_knobs: 
- parent_changed_symbols: none
- parent_last_proposal: none


Mode rationale:
- architecture defaults to deep mode
- top same-tier family contenders are within 0.0030 BPB

Ambiguous contenders:
- port_architecture_20260319_023023_5357 tier=smoke_mlx_local score=3.240207 gap=0.024907 bytes=2.95MB op=depth_sweep valid=true
- cod_architecture_20260319_024400_5344 tier=smoke_mlx_local score=3.240342 gap=0.024942 bytes=2.96MB op=codex_proposal valid=true
- port_architecture_20260319_022832_5187 tier=smoke_mlx_local score=3.240459 gap=0.029559 bytes=4.73MB op=arch_slim_10x576_mlp1 tpl=arch_slim_10x576_mlp1 valid=true

Champion set:
- first_candidate tier=smoke_local score=1.224366 bytes=15.86MB valid=true
- port_architecture_20260319_023023_5357 tier=smoke_mlx_local score=3.240207 gap=0.024907 bytes=2.95MB op=depth_sweep valid=true
- rec_validate_compression_recipe_warmdown tier=smoke_mlx_local score=3.045570 gap=0.018170 bytes=5.97MB op=recombine valid=true
- port_training_recipe_20260319_025140_4806 tier=smoke_mlx_local score=2.808115 gap=0.006215 bytes=6.37MB op=embed_lr_sweep valid=true

Broader family history:
- cod_architecture_20260319_024400_5344 tier=smoke_mlx_local score=3.240342 gap=0.024942 bytes=2.96MB op=codex_proposal valid=true
- port_architecture_20260319_023023_5357 tier=smoke_mlx_local score=3.240207 gap=0.024907 bytes=2.95MB op=depth_sweep valid=true
- port_architecture_20260319_022908_5031 tier=smoke_mlx_local score=3.241801 gap=0.034401 bytes=6.36MB op=mlp_sweep valid=true
- port_architecture_20260319_022832_5187 tier=smoke_mlx_local score=3.240459 gap=0.029559 bytes=4.73MB op=arch_slim_10x576_mlp1 tpl=arch_slim_10x576_mlp1 valid=true
- cod_architecture_20260319_022421_4423 tier=smoke_mlx_local score=3.246188 gap=0.024388 bytes=2.83MB op=codex_proposal valid=true
- port_architecture_20260319_020016_7076 tier=smoke_mlx_local score=3.246039 gap=0.024239 bytes=2.83MB op=mlp_sweep valid=true

Parent delta excerpt:
```diff
no diff from source
```


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
- Implement one promising improvement for the architecture family.
- Prefer a small but real code change over commentary.
- Do not touch files outside this candidate workspace.
- Use the context above to avoid duplicating already-tried weak ideas.

Final response:
- Briefly state the hypothesis.
- Mention the exact files you changed.
- Mention expected upside and main risk.
