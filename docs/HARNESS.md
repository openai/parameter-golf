# Parameter Golf Harness

Last updated: 2026-03-18

## Goal

The harness exists to make the experiment loop explicit and repeatable:

1. read prior runs
2. derive the next hypothesis
3. execute the run in an isolated directory
4. parse the trainer outputs
5. persist structured history
6. append a human-readable journal entry

The first version started parameter-first.
It can now mutate both environment-level training settings and a small deterministic set of trainer-code policies.
Code mutations are always materialized into run-local trainer copies under `lab/runs/<experiment_id>/`; the harness never rewrites the repo source files in place.

The current v2 direction is:

- make run validity explicit before the planner trusts a result
- attach every planned run to a hypothesis family and lineage
- separate cheap scout evidence from stronger challenge-relevant evidence
- prepare the planner for later promotion/funnel logic without turning v1 into a giant rewrite

## Trust layer

The harness now distinguishes between "a run happened" and "a run produced planner-worthy evidence".

Key trust fields:

- `run_state`: lifecycle state such as `dry_run`, `blocked`, `failed_pre_train`, `failed_mid_run`, `completed`
- `metrics_valid`: whether the parser saw enough real trainer signal to trust the metrics
- `planner_eligible`: whether autonomous planning is allowed to learn from the run
- `evidence_tier`: one of `record_import`, `challenge_record`, `challenge_candidate`, `subset`, `smoke`, `invalid`
- `inherits_parent_code_mutation`: whether an env-only follow-up is intentionally continuing the best parent run's code policy

Important rule:

- only planner-eligible runs with validated metrics may steer the next autonomous experiment
- dry runs, blocked runs, and invalid parses are still logged, but only as context

## Minimal funnel

The harness now has a lightweight funnel instead of a full scheduler:

- `scout`: cheap local signal-finding runs
- `confirm`: one confirmation pass for a promising local winner
- `candidate`: fuller non-record style profile when available
- `record_rehearsal`: challenge-like profile when the environment is genuinely ready

Current behavior is intentionally small:

- if the latest planner-eligible run looks like a winner, the harness plans one `confirm_winner` follow-up
- if that confirmation still looks strong, the harness plans one `neighbor_probe`
- after that it falls back to normal candidate generation
- env-only follow-ups inherit the best planner-eligible parent code mutation unless the planner explicitly chooses a different code mutation

## Layout

- `harness/`: code for planning, running, parsing, and journaling
- `lab/experiments.jsonl`: machine-readable run history
- `lab/runs/<experiment_id>/`: per-run working directories, specs, stdout, summaries, and artifacts
- [`docs/EXPERIMENT_JOURNAL.md`](/Users/kevin/Code/ParameterGolf_OAI/docs/EXPERIMENT_JOURNAL.md): human-facing run journal

## Why both JSONL and Markdown exist

The JSONL history is for the harness.
The Markdown journal is for us.

If the harness only wrote Markdown, planning would be brittle.
If it only wrote JSON, we would lose the running narrative and decision trail.

The JSONL history now also carries the first research-planning primitives:

- planner eligibility
- evidence tier
- run state / failure stage
- hypothesis family
- lineage id
- expected upside / risk / kill / promotion notes

## Current profiles

### `mlx_smoke`

Purpose:

- fast local loop on Apple Silicon
- uses `train_gpt_mlx.py`
- defaults to the checked-in local dataset/tokenizer
- starts from the short smoke-style setup

### `torch_single_gpu_smoke`

Purpose:

- local CUDA smoke loop
- uses `train_gpt.py`
- intended for later GPU-equipped environments

### `torch_record_8gpu`

Purpose:

- real record-track orchestration target
- models the 10-minute `8xH100` style run shape
- preflight-blocks when the environment is not actually challenge-capable

### `torch_nonrecord_8gpu`

Purpose:

- longer 8-GPU exploratory runs
- useful for frontier scouting before compressing ideas back into record-track constraints

## Commands

Bootstrap existing public records into structured history:

```bash
python3 -m harness bootstrap
```

Inspect recent structured history:

```bash
python3 -m harness inspect
```

Run a challenge/data/environment preflight for the next planned spec:

```bash
python3 -m harness preflight --profile mlx_smoke
```

Show which profiles are actually executable and which are challenge-ready:

```bash
python3 -m harness doctor
```

Run a built-in harness self-check:

```bash
python3 -m harness selfcheck
```

Plan the next run without executing it:

```bash
python3 -m harness plan --profile mlx_smoke
```

Run one autonomous experiment:

```bash
python3 -m harness run --profile mlx_smoke
```

Run a short autonomous loop:

```bash
python3 -m harness loop --profile mlx_smoke --max-runs 3
```

Force a quick verification override:

```bash
python3 -m harness run --profile mlx_smoke --override ITERATIONS=5 --override TRAIN_BATCH_TOKENS=8192
```

Force a specific code mutation without launching real training:

```bash
python3 -m harness run --profile mlx_smoke --code-mutation quant_clip_tighter --dry-run
```

## Planner behavior in v1

The planner is currently heuristic, not model-based.

It does four useful things already:

1. establishes a clean baseline when no comparable history exists
2. keeps planning profile-local, so smoke runs are not confused with imported leaderboard records
3. prefers stability probes when the quantization gap is large
4. avoids exact duplicate environment configurations
5. runs with a hard per-profile timeout guardrail
6. blocks obviously invalid runs with a preflight before burning compute
7. kills runs that stop producing output for too long
8. refuses challenge-only profiles until the environment is genuinely challenge-ready
9. launches torch jobs through the same preferred Python that preflight validated
10. annotates every planned run with explicit hypothesis metadata so lines of investigation can be tracked over time
11. confirms promising winners once before spending more search budget nearby

Current mutation families:

- tied embedding LR
- matrix LR
- scalar LR
- QK gain
- gradient clipping
- warmdown schedule
- iteration budget
- quantization clip percentile
- quantization scale dtype
- float-passthrough threshold

## Isolation model

Every harness run gets its own working directory under `lab/runs/`.

This matters because:

- trainer logs stay attached to the exact run
- artifacts do not overwrite each other
- run-level `spec.json`, `env.json`, `stdout.txt`, and `summary.json` stay together

## Essential guardrails in v1.1

- preflight before launch
- comparability labeling
- trust-layer validation before planner reuse
- challenge-readiness gating for challenge profiles
- per-profile hard timeout
- no-progress idle timeout
- planner dedupe across prior profile runs
- one-command doctor view across all profiles
- selfcheck command using real record logs

## Current limits

The current harness still does not:

- perform free-form repo-wide code edits
- auto-branch or commit
- synthesize arbitrary new architecture ideas without an explicit deterministic mutation recipe
- auto-rank multi-file code rewrites against small policy mutations

Those are deliberate v2 features.
For now, the important thing is that the feedback loop is real, persistent, and trustworthy.

## v2 patch line

The first v2 patch series is intentionally small:

1. harden trust in run outputs
2. add hypothesis metadata to planned specs
3. keep invalid or non-planner-eligible runs out of comparable planning history
4. surface research lineage in inspect/journal output
