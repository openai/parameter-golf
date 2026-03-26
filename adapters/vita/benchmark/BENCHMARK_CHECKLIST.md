# VITA -> Parameter-Golf Benchmark Checklist (Non-Claiming Scaffold)

Status policy:
- Until the real parameter-golf pipeline is executed under challenge-like constraints, this workstream is NON-CLAIMING.
- Packaging quality is not benchmark validity.

## 1) What counts as a real benchmark run

A run is considered benchmark-valid only when all required items below are satisfied.

Required environment + protocol:
- [ ] Real benchmark pipeline executed (parameter-golf training/eval path, not only adapter scripts).
- [ ] Target task/metric path is the challenge path (FineWeb + val_bpb flow), not surrogate metrics.
- [ ] Hardware/protocol documented (including whether run was on 8xH100 SXM or not).
- [ ] Training wallclock policy documented and enforced.
- [ ] Evaluation wallclock policy documented and enforced.

Required measured outputs:
- [ ] Training log captured (`train.log`), with exact command and env vars used.
- [ ] Final evaluation line captured from benchmark path (e.g., exact printed metric line).
- [ ] Artifact size accounting captured (code bytes + compressed model bytes).
- [ ] Submission metadata generated from benchmark outputs (not hand-entered claims).

Required reproducibility/claim hygiene:
- [ ] Clear statement of what is and is not comparable to leaderboard entries.
- [ ] If making competitive/SOTA claim: statistical and reproducibility evidence included.
- [ ] If not challenge-conform hardware/protocol: explicitly marked as non-record/non-claiming.

## 2) Hard no-claim rules

Do NOT claim any of the following until real benchmark run evidence exists:
- leaderboard competitiveness
- challenge-equivalent performance
- SOTA or record improvement
- hardware-comparable efficiency

## 3) Expected benchmark run outputs

Store benchmark-run artifacts under:
- `adapters/vita/benchmark/runs/<run_tag>/`

Expected files:
- `benchmark_manifest.json` (run plan + provenance + blockers)
- `benchmark_claims_guard.json` (explicit non-claiming state)
- `benchmark_commands.sh` (exact commands to run benchmark path)
- `expected_outputs.md` (required output files and acceptance checks)
- `notes.md` (operator notes and deviations)
- `evidence/` directory with real run outputs once executed:
  - `train.log`
  - `eval.log` (or combined log with identified eval segment)
  - `submission.json`
  - `artifact_sizes.json`
  - `environment.json`

## 4) Acceptance gate to switch from NON-CLAIMING -> CLAIM-ELIGIBLE

Switch allowed only when:
- [ ] Real benchmark logs exist in `evidence/`
- [ ] Exact benchmark metric extracted from those logs
- [ ] Artifact-size and protocol constraints verified
- [ ] Claim language updated with explicit comparability limits
