# VITA Adapter Scaffold (parameter-golf style packaging)

This scaffold mirrors the `parameter-golf` evidence/packaging style for VITA Objective-B runs:
- explicit run summary,
- machine-readable `submission.json`,
- compact `results.tsv`,
- reproducible command block and artifact pointers.

It does NOT change challenge scoring in this repository.
It only provides a structured packaging layer for your VITA 090 optimization outputs.

## Inputs expected (from vita-autoresearch)

- O1 report JSON (default):
  - `/Users/ever/Documents/GitHub/vita-autoresearch/pass2/targeted/optimization/o1/reports/o1_results.json`
- O2 report JSON (default):
  - `/Users/ever/Documents/GitHub/vita-autoresearch/pass2/targeted/optimization/o2/reports/o2_results.json`

## What gets generated

`adapters/vita/out/<tag>/`
- `submission.json` (leaderboard/submission-shaped metadata)
- `README.md` (human-readable evidence chain)
- `results.tsv` (ratio/accuracy sweep for the winner)
- `leaderboard_row.json` (single-row payload suitable for a table renderer)

## Verification gate (anti-drift)

You can enforce strict expectation checks before publish:

```bash
python3 adapters/vita/build_submission.py \
  --verify-only \
  --expected-winner C2 \
  --expected-max-ratio 0.7 \
  --expected-acc-at-max 87.27 \
  --expected-mean-band 87.3225 \
  --expected-confirm-acc 87.37
```

Behavior:
- exits `0` on full match (`VERIFY_OK`)
- exits `2` on mismatch (`VERIFY_FAIL`)
- in `--verify-only` mode, expected fields are required

## Benchmark-run scaffold (explicitly non-claiming)

A separate workstream is provided under:
- `adapters/vita/benchmark/`

Key files:
- `BENCHMARK_CHECKLIST.md`
- `MAPPING_090_C2_TO_BENCHMARK.md`
- `init_benchmark_run.py`

Initialize a benchmark attempt scaffold:

```bash
python3 adapters/vita/benchmark/init_benchmark_run.py \
  --run-tag 090-c2-benchmark-attempt-01 \
  --source-repo-id 090 \
  --source-config-id C2
```

This creates a non-claiming run directory with manifest, claims guard, command placeholders, and expected-output checks.

## Usage

From repo root:

```bash
python3 adapters/vita/build_submission.py \
  --tag 090-o2 \
  --author ever \
  --github-id ever-oli
```

Optional overrides:

```bash
python3 adapters/vita/build_submission.py \
  --o1 /path/to/o1_results.json \
  --o2 /path/to/o2_results.json \
  --repo-id 090 \
  --repo-name "SFW Once-for-All Pruning" \
  --floor 74.5 \
  --tag custom-run
```

## Notes

- This adapter treats score as Objective-B tuple:
  1) max prune ratio passing floor,
  2) accuracy at that ratio,
  3) mean accuracy in 0.4-0.7 band.
- Default operating point is preserved from O2 (`prune_ratio=0.5` unless overridden by evidence).
