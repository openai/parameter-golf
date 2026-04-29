# Path B PPM-D Legality-Proof — Result Note

**Date:** 2026-04-29
**Artifact under audit:** `results/exp_1876_ppmd/prod_8gpu_s42v2/final_model.int6.ptz`
(15,975,706 bytes, exp_1876 PR #1876 + PPM-D mixture stack, seed 42)

## Summary

A defensible byte-level Path B evaluation of the exp_1876 PPM-D mixture
artifact has been completed on the first 8,000,000 validation tokens
(KNOWN_FIRST_8M_TOKEN_BYTES = 29,365,687 emitted target bytes).

**Result JSON:**
`results/exp_1876_ppmd/path_b_prod_8gpu_local_score/path_b_sliding_subset_8000000.json`

**Headline metrics (subset, audited):**

| Metric | Value |
| --- | --- |
| `mixture_bpb` | **1.5459** |
| `neural_only_bpb` | **1.5619** |
| `ppm_d_only_bpb` | **2.1184** |
| `denominator_match` | true (29,365,687 / 29,365,687) |
| `claim_ready` | true |
| `subset_tokens` | 8,000,000 |
| `runtime_seconds` (PPM-D scoring only) | 6,209.2 |

## Legality Finding

The previously reported PPM-D mixture number for this same artifact was
`0.99487 BPB`, computed via the in-source `_ppm_mixture_bpb` helper that
treats per-token neural logprobs as if they were per-byte logprobs and
mixes them with a token-level PPM-D context model. The audited Path B
byte-level mixture, which:

1. Marginalises neural logprobs over true emitted byte sequences via
   token-byte tries,
2. Models PPM-D over actual bytes with score-before-update,
3. Uses prefix-only (target-blind) lambda gating,
4. Normalises by the true emitted byte count,

reports **mixture_bpb = 1.5459** on the same model, on the same first 8M
validation tokens. The delta of approximately **+0.55 BPB** between the
two estimators on the same artifact is direct empirical confirmation of
the concern raised in openai/parameter-golf Issue #1872: the
`_ppm_mixture_bpb` figure used in the original PR #1876+PPM-D submission
is not a valid bits-per-byte score for the contest.

The prior post-TTT neural-only contest BPB for the same artifact
(reported as 1.08122 with TTT) was computed by a completely different
evaluation path (full-val sliding eval with token-level NLL→BPB
conversion plus TTT). That number is not directly comparable to the
audited Path B subset numbers above and is not invalidated by this
finding.

## Acceptance Gates Passed

* `denominator_match` = true (8M-token subset, expected 29,365,687 bytes)
* All 8 rank shards present, sha256-hashed, manifest written
* PPM-D order 5, score-before-update preserved
* Lambda gating prefix-only (target byte never inspected before scoring)
* Verified `fast_score_ppmd_stream` agrees with the reference
  `score_ppmd_stream` to 1e-12 on the first 5,000 records before
  running on the full stream
* Zero `warnings`, no `error`

## Scope and Caveats

* **Subset only.** This eval covers the first 8M validation tokens
  (29,365,687 emitted bytes), not the full 151,078,222-byte validation
  split. A future authorized 8×H100 production run with
  `--subset-tokens 0` (full) would be required for a full-val claim.
  The first-8M subset is contiguous in absolute token order and uses
  the documented `KNOWN_FIRST_8M_TOKEN_BYTES` denominator.
* **No TTT.** Path B sliding eval is the audit, not the contest score.
  The `--eval-kind ttt` path remains intentionally `NotImplementedError`
  in `scripts/eval_path_b_ppmd.py`.
* **8×H100 underutilised for subset eval.** With contiguous
  rank-window slicing, only ranks 0–1 actually score windows in
  `[0, 8M)`; ranks 2–7 produced empty NPZ shards. This was acceptable
  here because the goal was the audit, not throughput.

## Operational Notes

The 8×H100 production windowing pass (rank shards) was performed on
RunPod under the HTTP-bootstrap path documented in `AGENTS.md`, with
node-side timed shutdown. After the rank shards were retrieved to this
HPC, the merge + PPM-D scoring step was performed locally with
`module load conda/py312/1.1` to obtain a usable `python3.12`
interpreter. The tight inner loop in `score_ppmd_stream` calls
`PPMDByteModel.distribution()` once per byte (the original code path
called it twice via `ppmd_prefix_lambda`); the local scorer
(`/tmp/fast_score.py`) collapses that to a single call and was verified
to match the reference exactly before being used to produce the
headline numbers.

Total RunPod cost for the windowing pass: approximately $4 (one
successful 8-GPU production attempt + earlier 1-GPU rehearsals).
PPM-D scoring on this HPC: free.

## Next Authorized Steps (NOT executed)

* If a full-val (151,078,222-byte) audited mixture BPB is desired for
  publication, run `scripts/eval_path_b_ppmd.py` on 8×H100 with
  `--subset-tokens 0`. Expected windowing wallclock on 8 ranks: similar
  to the subset run; expected PPM-D scoring wallclock locally on
  Python 3.12: approximately 8–9 hours linearly extrapolated.
* Any external claim that the exp_1876 + PPM-D stack achieves
  sub-1.0 BPB on the contest metric should be retracted or qualified
  with the audited 1.546 mixture BPB number from this report.
