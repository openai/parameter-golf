# Repo Working Notes

Last updated: 2026-03-18

## Purpose

Local repo guide for working on Parameter Golf with Codex.
This is a compact operational mirror of the active repo instructions plus the local harness/doc structure.

## Ground Rules

- No guessing on OpenAI or challenge details. Verify from repo code/docs or official sources.
- Keep diffs small and reviewable.
- Prefer root-cause fixes over one-off patches.
- Do not add secrets, tokens, private URLs, or credentials anywhere.
- When touching challenge behavior, think in terms of post-quant roundtrip score and total artifact bytes, not raw training loss alone.

## Must-Read Files

- [`docs/CHALLENGE_REFERENCE.md`](/Users/kevin/Code/ParameterGolf_OAI/docs/CHALLENGE_REFERENCE.md)
- [`docs/EXPERIMENT_JOURNAL.md`](/Users/kevin/Code/ParameterGolf_OAI/docs/EXPERIMENT_JOURNAL.md)
- [`docs/HARNESS.md`](/Users/kevin/Code/ParameterGolf_OAI/docs/HARNESS.md)
- [`README.md`](/Users/kevin/Code/ParameterGolf_OAI/README.md)
- [`data/README.md`](/Users/kevin/Code/ParameterGolf_OAI/data/README.md)

## Current Reality

- The checked-in local dataset is a smoke subset, not the full published training prefix.
- The main training scripts are [`train_gpt.py`](/Users/kevin/Code/ParameterGolf_OAI/train_gpt.py) and [`train_gpt_mlx.py`](/Users/kevin/Code/ParameterGolf_OAI/train_gpt_mlx.py).
- The local autonomous harness lives in `harness/`.
- Harness state and run artifacts live under `lab/`.
- Harness code mutations must stay run-local under `lab/runs/` and must not rewrite repo trainer files in place.

## Experiment Discipline

- Every meaningful run should end up in the structured harness history or the Markdown journal.
- Only planner-eligible runs with validated metrics may influence autonomous planning; dry runs, blocked runs, and invalid parses are context only.
- Preserve the research trail for harness-planned runs: hypothesis family, lineage id, expected upside, risk level, kill criteria, promotion rule, evidence tier.
- When a planner follow-up does env tuning around a successful code-mutated parent, carry that parent code mutation forward unless the next candidate explicitly replaces it.
- Keep apples-to-apples comparisons explicit:
  - dataset/tokenizer
  - profile/track
  - quantization path
  - wallclock regime
  - code snapshot
- Imported historical `records/` entries are context, not automatically comparable to local smoke runs.

## Harness Workflow

- Bootstrap historical records with `python3 -m harness bootstrap`
- Inspect recent structured history with `python3 -m harness inspect`
- Plan with `python3 -m harness plan --profile mlx_smoke`
- Preflight a planned run with `python3 -m harness preflight --profile mlx_smoke`
- Check launchability and challenge readiness with `python3 -m harness doctor`
- Run harness self-checks with `python3 -m harness selfcheck`
- Execute with `python3 -m harness run --profile mlx_smoke`
- Loop with `python3 -m harness loop --profile mlx_smoke --max-runs N`

## Update Rule

Update this file whenever one of these changes:

- repo workflow
- harness commands or layout
- key challenge-memory file locations
- experiment logging conventions
