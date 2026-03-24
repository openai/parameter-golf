# Mapping: VITA 090/C2 -> Parameter-Golf Benchmark Inputs

This document is a planning map, not a claim.

## A) Source artifacts (already available)

From VITA optimization outputs:
- O2 report JSON:
  - `/Users/ever/Documents/GitHub/vita-autoresearch/pass2/targeted/optimization/o2/reports/o2_results.json`
- Winner config: `C2`
- Winner metrics (objective-B side):
  - train test accuracy: 87.28
  - sweep 0.4/0.5/0.6/0.7: 87.33 / 87.37 / 87.32 / 87.27
  - confirm@0.5: 87.37

Available packaged evidence (adapter path):
- `adapters/vita/out/<tag>/submission.json`
- `adapters/vita/out/<tag>/README.md`
- `adapters/vita/out/<tag>/results.tsv`
- `adapters/vita/out/<tag>/leaderboard_row.json`

## B) Benchmark-side required inputs (parameter-golf path)

Parameter-golf benchmark expects language-model challenge path inputs and flow, including:
- benchmark training/eval scripts (`train_gpt.py` path)
- FineWeb/tokenizer path
- benchmark metric extraction (`val_bpb` path)
- artifact-size accounting in challenge style

## C) Mapping status

1) Directly reusable now:
- provenance metadata (winner id, selection rationale, evidence chain)
- run manifest scaffolding
- packaging structure style and metadata hygiene

2) Needs adaptation (major):
- model/task mismatch: VITA 090/C2 is CIFAR pruning workflow; parameter-golf benchmark is LM compression metric path
- metric mismatch: objective-B prune/accuracy tuple != challenge `val_bpb`
- training/eval codepath mismatch: requires benchmark script execution and challenge data pipeline

3) Needs real execution evidence:
- benchmark logs from actual benchmark run
- exact final benchmark metric line(s)
- challenge-style artifact-size accounting from produced benchmark artifacts

## D) Non-claim statement

Until adaptation + real benchmark execution complete, 090/C2 can only be described as:
- selected and optimized in the VITA objective-B workflow,
- packaged in parameter-golf-inspired format,
- NOT benchmarked on actual parameter-golf challenge path.

## E) Minimal adaptation plan (implementation workstream)

- Define a benchmark candidate implementation path in parameter-golf repo (separate branch/workstream).
- Implement/run benchmark script path with explicit commands.
- Capture evidence files under `adapters/vita/benchmark/runs/<run_tag>/evidence/`.
- Only then evaluate comparability/claim status.
