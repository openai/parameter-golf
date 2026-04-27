# Frontier Runs

The repo is now split into three explicit lanes:

- `stable`: submission-facing, reviewer-friendly, promotion-eligible.
- `challenger`: opt-in only, manual review required, never submission-safe by default.
- `research_only`: separate runner under `research_only/`, never launched from `research/run.py`.

## Current Branch Of Record

The default branch-of-record is the SP8192 mainline family:

- `sp8192_mainline_base`
- `sp8192_mainline_recur345_par7`
- `sp8192_mainline_recur345_par7_qk525`
- `sp8192_mainline_recur345_par7_qk525_ttt`
- `sp8192_mainline_submit_safe`

Use these for stable-lane work. They are the only presets intended for automatic promotion and final submission packaging.

## Stable Lane

Stable-lane rules:

- no prefix matcher
- no pre-quant TTT
- no challenger-only hooks
- canonical reporting only
- `submission_safe=true` is possible only here

Recommended commands:

```bash
python3 research/run.py --preset sp8192_mainline_base --scale smoke --run-name sp8192_mainline_base_smoke --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset sp8192_mainline_recur345_par7_qk525_ttt --scale half_run --run-name sp8192_mainline_half --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset sp8192_mainline_submit_safe --scale submit_safe --run-name sp8192_mainline_submit --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

Quantization and tokenizer surfaces:

- `quant_grouped_sdclip`
- `quant_calib_mixed`
- `quant_embed_split_policy`
- `tok_sp8192_clean`
- `tok_sp7680_clean`
- `tok_sp7168_clean`

## Challenger Lane

Challenger-lane rules:

- must be launched with `--allow-challenger`
- always emits `manual_review_required=true`
- never emits `submission_safe=true`
- always writes a `rule_audit.{json,txt}` artifact

Available challenger presets:

- `challenger_prefix_matcher`
- `challenger_prefix_matcher_ttt`
- `challenger_prequant_ttt`
- `challenger_prefix_matcher_prequant_ttt`

Example:

```bash
python3 research/run.py --preset challenger_prefix_matcher --scale half_run --run-name challenger_prefix_half --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100 --allow-challenger
```

## Run Artifacts

Every completed run now records:

- training-best validation BPB
- official post-quant submission BPB
- artifact bytes and remaining headroom
- train time and official eval time
- quant gap when available
- stable/challenger lane metadata
- deltas versus the `2026-04-11` merged and live leaderboard bars

Use these to decide whether to promote, discard, or escalate a branch.

## Submission Readiness

Check the latest stable candidate:

```bash
python3 scripts/submission_readiness.py --latest --family frontier --lane stable --require-submission-safe
```

Backfill an older run:

```bash
python3 scripts/submission_readiness.py --run-dir research/results/runs/<timestamp_run_name> --rewrite
```

## Legacy Appendix

The older cache / Dirichlet / frontier-control stack is still present but explicitly demoted to `legacy`:

- `control_verified_sota`
- `sota_plus_ngram7`
- `sota_plus_ppm_multiorder`
- `sota_plus_ppm_entropy_fixed`
- `sota_plus_ppm_entropy_order_adaptive`
- `sota_plus_ppm_dirichlet`
- `sota_plus_ppm_dirichlet_submit`
- `xsaall_fullgptq_prune_plus_cache`
- `rotaryfix_bigram3072_legalttt`

These remain useful as references and rollback points, but they are not the default branch-of-record for new leaderboard work.
