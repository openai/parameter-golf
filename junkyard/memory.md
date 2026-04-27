# Lab Memory: Operating Procedures

## Purpose
This file is the canonical operating reference for experiment safety, reproducibility, and promotion decisions.

## Non-Negotiable Rules
1. Any file that has been run to produce a result is immutable.
2. Never edit a run-tested file in place. Always copy forward to a new path.
3. `SOTA` artifacts are vault references and remain immutable.
4. Active development happens on copied variants (for example: stripped or ablation branches).
5. One hypothesis per ablation variant unless explicitly labeled as a combo test.

## Baseline and Variant Policy
1. Keep one frozen baseline of record.
2. For each new test, create a new variant directory/file from the current approved base.
3. Variant names must encode intent (for example: `ablate_loader_cache2`, `ablate_muon_ns3`).
4. If parity between cleaned/refactored code and baseline is not proven, do not promote the cleaned version.

## Required Experiment Record (Every Run)
Record these fields for each run:
- experiment_id
- parent_artifact (exact source file/path copied from)
- changed_files (full paths)
- hypothesis
- ablation description
- test command
- seed(s)
- steps / wallclock cap
- hardware (GPU count/type)
- dataset path + tokenizer path
- key env vars
- metrics (primary + secondary)
- output artifact size

## Pre-Run Checklist
1. Confirm run target points to a new copied file, not a previously run file.
2. Confirm only in-scope files changed.
3. Confirm seed/steps/config match comparison policy.
4. Confirm eval mode matches intended signal test (for example: no winddown or no final eval when requested).

## Post-Run Checklist
1. Freeze the exact file(s) used by the run.
2. Append run summary and metrics.
3. Mark outcome: win, neutral, or loss vs baseline.
4. If win is repeatable, promote by copying into a new approved baseline path (never mutate old baseline).

## Promotion Gates
1. Parity gate: refactor/cleanup must match baseline within agreed tolerance before becoming active base.
2. Performance gate: variant must beat baseline on agreed metric under comparable settings.
3. Repro gate: winner must reproduce across agreed seeds/reruns.

## Scope Lock Procedure
Before editing:
1. State exact files to touch.
2. If any out-of-scope file is needed, stop and re-approve scope.
3. After editing, verify changed file list contains only intended new variant paths.

## Fast-Fail Diagnostics
Stop and investigate immediately when any of these happen:
- metric drift inconsistent with prior ablations
- unexpected artifact size change
- unexpected runtime/throughput jump
- data path or tokenizer mismatch
- world size / grad accumulation mismatch

## Anti-Regression Guardrails
1. Prefer scripted checks that fail if immutable files are modified.
2. Keep a machine-readable immutable registry where practical.
3. Treat environment changes as explicit experiments, not hidden background changes.

## Decision Principle
If a result is not reproducible, attributable, and comparable, it does not qualify for promotion.
