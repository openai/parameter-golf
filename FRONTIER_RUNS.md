# Frontier Runs

This repo is now wired for the current legal frontier presets through `research/run.py`.

For RunPod / PyTorch-image setup, use [CLOUD_SETUP.md](CLOUD_SETUP.md) before launching these presets.

Canonical control:

- `control_verified_sota`
- Trainer: `train_gpt_frontier_control.py`
- This is the March 23 verified control family: LeakyReLU^2 + legal score-first TTT + Parallel Muon.
- Treat this as the benchmark reference for all serious 8xH100 comparisons.

Important legality and reproducibility notes:

- Cache presets use a strict score-then-commit interface from `frontier_cache.py`. A scored segment cannot be committed early, and the next segment cannot be scored until the prior one is committed.
- TTT remains score-first: each chunk is scored before any chunk-local adaptation step.
- Cache-enabled distributed eval uses a canonical rank-0 evaluation path, then broadcasts metrics. This is slower than fully sharded eval, but it keeps cache semantics deterministic and legal.
- `artifact_bytes` are measured from the counted code bundle plus the exported compressed model blob. The checkpoint helper in `frontier_checkpoint.py` is included in `code_bytes`.
- Exact frontier presets remain CUDA-only. For Apple Silicon smoke work, use the existing MLX proxy presets, not these final frontier trainers.

## Configs

| Preset | What it tests | Risk |
| --- | --- | --- |
| `control_verified_sota` | Verified control branch, no eval cache | green |
| `sota_plus_ngram7` | Verified control + deterministic 7-gram backward-looking cache | green |
| `sota_plus_ppm_multiorder` | Verified control + deterministic multi-order PPM-style backoff cache | yellow |
| `sota_plus_ppm_entropy_fixed` | PPM backoff + fixed entropy-gated eval-time cache mixing | yellow |
| `sota_plus_ppm_entropy_order_adaptive` | PPM backoff + order-adaptive entropy-gated eval-time cache mixing | yellow |
| `sota_plus_ppm_dirichlet` | PPM backoff + single-pass Dirichlet posterior predictive mixing | yellow |
| `xsaall_fullgptq_prune_plus_cache` | XSA-all + full GPTQ + selective pruning + causal cache | yellow |
| `rotaryfix_bigram3072_legalttt` | RotaryFix + BIGRAM3072 + legal score-first TTT | green |

## Commands

`control_verified_sota`

```bash
python3 research/run.py --preset control_verified_sota --scale smoke --run-name control_verified_sota_smoke_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset control_verified_sota --scale half_run --run-name control_verified_sota_half_run_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset control_verified_sota --scale full_run --run-name control_verified_sota_full_run_s1337 --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

`sota_plus_ngram7`

```bash
python3 research/run.py --preset sota_plus_ngram7 --scale smoke --run-name sota_plus_ngram7_smoke_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset sota_plus_ngram7 --scale half_run --run-name sota_plus_ngram7_half_run_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset sota_plus_ngram7 --scale full_run --run-name sota_plus_ngram7_full_run_s1337 --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

`sota_plus_ppm_multiorder`

```bash
python3 research/run.py --preset sota_plus_ppm_multiorder --scale smoke --run-name sota_plus_ppm_multiorder_smoke_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset sota_plus_ppm_multiorder --scale half_run --run-name sota_plus_ppm_multiorder_half_run_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset sota_plus_ppm_multiorder --scale full_run --run-name sota_plus_ppm_multiorder_full_run_s1337 --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

`sota_plus_ppm_entropy_fixed`

```bash
python3 research/run.py --preset sota_plus_ppm_entropy_fixed --scale smoke --run-name sota_plus_ppm_entropy_fixed_smoke_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset sota_plus_ppm_entropy_fixed --scale half_run --run-name sota_plus_ppm_entropy_fixed_half_run_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset sota_plus_ppm_entropy_fixed --scale full_run --run-name sota_plus_ppm_entropy_fixed_full_run_s1337 --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

`sota_plus_ppm_entropy_order_adaptive`

```bash
python3 research/run.py --preset sota_plus_ppm_entropy_order_adaptive --scale smoke --run-name sota_plus_ppm_entropy_order_adaptive_smoke_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset sota_plus_ppm_entropy_order_adaptive --scale half_run --run-name sota_plus_ppm_entropy_order_adaptive_half_run_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset sota_plus_ppm_entropy_order_adaptive --scale full_run --run-name sota_plus_ppm_entropy_order_adaptive_full_run_s1337 --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

`sota_plus_ppm_dirichlet`

```bash
python3 research/run.py --preset sota_plus_ppm_dirichlet --scale smoke --run-name sota_plus_ppm_dirichlet_smoke_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset sota_plus_ppm_dirichlet --scale half_run --run-name sota_plus_ppm_dirichlet_half_run_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset sota_plus_ppm_dirichlet --scale full_run --run-name sota_plus_ppm_dirichlet_full_run_s1337 --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

`xsaall_fullgptq_prune_plus_cache`

```bash
python3 research/run.py --preset xsaall_fullgptq_prune_plus_cache --scale smoke --run-name xsaall_fullgptq_prune_plus_cache_smoke_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset xsaall_fullgptq_prune_plus_cache --scale half_run --run-name xsaall_fullgptq_prune_plus_cache_half_run_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset xsaall_fullgptq_prune_plus_cache --scale full_run --run-name xsaall_fullgptq_prune_plus_cache_full_run_s1337 --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

`rotaryfix_bigram3072_legalttt`

```bash
python3 research/run.py --preset rotaryfix_bigram3072_legalttt --scale smoke --run-name rotaryfix_bigram3072_legalttt_smoke_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile local_cuda
python3 research/run.py --preset rotaryfix_bigram3072_legalttt --scale half_run --run-name rotaryfix_bigram3072_legalttt_half_run_s1337 --seed 1337 --nproc-per-node 1 --gpu-profile 1xh100
python3 research/run.py --preset rotaryfix_bigram3072_legalttt --scale full_run --run-name rotaryfix_bigram3072_legalttt_full_run_s1337 --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100
```

Resume an interrupted run:

```bash
python3 research/run.py --resume-run-dir research/results/runs/<timestamp_run_name>
```

Compare runs:

```bash
python3 research/compare_runs.py --family frontier --status all --limit 20
python3 research/compare_runs.py --family frontier --scale half_run --status all --json
```

Preflight an expensive distributed launch without starting training:

```bash
python3 research/run.py --preset sota_plus_ppm_entropy_fixed --scale full_run --run-name sota_plus_ppm_entropy_fixed_full_run_preflight --seed 1337 --nproc-per-node 8 --gpu-profile 8xh100 --preflight-only
```

Inspect progress:

```bash
LATEST_RUN="$(ls -td research/results/runs/* | head -n 1)"
cat "$LATEST_RUN/run_summary.json"
tail -f "$LATEST_RUN/launcher.log"
```

Inspect byte budget for a completed run:

```bash
LATEST_RUN="$(ls -td research/results/runs/* | head -n 1)"
cat "$LATEST_RUN/byte_budget.txt"
cat "$LATEST_RUN/legality_note.txt"
cat "$LATEST_RUN/submission_readiness.txt"
```

Check submission-readiness consistency or backfill canonical metric fields for an older run:

```bash
python3 scripts/submission_readiness.py --latest --family frontier
python3 scripts/submission_readiness.py --run-dir research/results/runs/<timestamp_run_name> --rewrite
```

## Current H100 Policy

- Preserve the stabilized H100 path. Keep FlashAttention working and do not reintroduce `torch.compile` into the frontier trainers.
- Treat `sota_plus_ppm_entropy_fixed` as the default branch for serious runs.
- Current branch of record: `sota_plus_ppm_entropy_fixed` at `legal_ttt val_bpb = 2.506169`.
- For one higher-upside legal 8xH100 compliance attempt, test `sota_plus_ppm_dirichlet` against the same stabilized trainer path before changing anything else.
- For cache sweeps, trust the resolved startup `causal_cache:` line in the trainer log, or the launcher's `resolved_cache:` dry-run output. Run name and shell env alone are not sufficient.
- Avoid tiny entropy-only knob sweeps unless they are bundled with a larger hypothesis.

## Promotion Ladder

Stage A: promotion / validation

- Reconfirm `sota_plus_ppm_entropy_fixed` only when needed for confidence or before a larger promotion.
- Use 1xH100 for surgical checks only.
- Keep weaker branches parked unless they carry a new structural change.

Stage B: stronger runs

- Promote the best fixed-entropy branch to longer runs on 1xH100 or 2-3xH100 if supported by repo infrastructure.
- Use these runs to test train-time / wallclock scaling on the winning branch, not to branch out widely.
- Prioritize adaptation schedule changes, evaluation-side strategy changes, and other larger hypothesis-driven updates over minor entropy-threshold tuning.

Stage C: leaderboard path

- Only the strongest promoted branch should move to 8xH100 scale.
- Reserve 8xH100 for runs with a clear promotion signal from smaller-scale results.

## Practical Priorities

Prioritize:

- train-time / wallclock scaling on `sota_plus_ppm_entropy_fixed`
- adaptation schedule changes
- evaluation-side strategy changes
- larger hypothesis-driven changes rather than minor entropy sweeps

De-prioritize:

- repeated center-only sweeps
- repeated slope-only sweeps
- re-testing known weaker branches without a new reason

## Decision Rule

- If a new half-run does not beat `2.506169` by a meaningful margin, do not promote it.
- If a variant only ties within noise, keep `sota_plus_ppm_entropy_fixed` as the branch of record.
- Promote only changes that show either a clear metric gain or a strong scaling rationale.

## Kill Criteria

Stop or reject a branch immediately if any of these fire:

- `legality_note.json` status is not `legal`
- `byte_budget.json` reports `artifact_bytes_measured > 16000000`
- Cache branch `eval_time_seconds` is worse than the control by more than about 1.5x at the same scale
- Half-run `val_bpb` is worse than the half-run control by more than `0.005`
- Export fails, checkpoint resume fails, or the final eval label falls back to an incomplete/non-final metric

Soft yellow-branch kills:

- `xsaall_fullgptq_prune_plus_cache` has weak half-run signal and poor byte headroom at the same time
- `sota_plus_ppm_multiorder` is not clearly better than `sota_plus_ngram7`

## Outputs

Every completed run directory now includes:

- `run_spec.json`
- `result.json`
- `run_summary.json`
- `legality_note.json`
- `legality_note.txt`
- `byte_budget.json`
- `byte_budget.txt`
- `checkpoint_latest.pt` when checkpointing is enabled

For completed frontier runs, `training_best_val_bpb` and `training_best_val_loss` remain the best pre-export training metrics.

For official submission reporting, use `final_submission_metric_label`, `official_submission_metric_label`, `final_submission_bpb`, and `final_submission_loss`. `run_summary.json`, `result.json`, and `research/compare_runs.py` now converge on the same canonical submission metric.

Old completed run directories are not auto-backfilled. Regenerate or rerun the export/finalization path if you need those fields added to older artifacts.

The byte budget report is emitted automatically after a successful export and includes:

- `param_count`
- `raw_bytes_fp16_est`
- `raw_bytes_fp32_est`
- `post_quant_bytes_est`
- `exported_bytes_measured`
- `code_bytes_measured`
- `artifact_bytes_measured`
- `remaining_headroom_to_16MB`

## Recommendation

If you want one branch to get the next serious H100 spend, make it `sota_plus_ppm_entropy_fixed`.

Reason:

- it is the current best validated legal half-run branch in this repo snapshot
- it is clearly better than plain multiorder and current order-adaptive entropy results
- recent center / slope sweeps are near saturation, so the next gains should come from scaling or larger structural changes rather than more tiny entropy tuning

For one bounded legal 8xH100 experiment that changes only the cache mixer, use `sota_plus_ppm_dirichlet` and keep `sota_plus_ppm_entropy_fixed` as the rollback branch of record.
