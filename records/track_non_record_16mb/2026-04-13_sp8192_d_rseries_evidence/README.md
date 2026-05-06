# SP8192 D/R-Series Evidence Package

This folder is a non-record submission package for the SP8192 `D` branch and the 2026-04-09 R-series sweep.

It packages the strongest evidence line from this branch in a reviewer-friendly shape:

- canonical base: the 5-seed `D` bundle built from the SP8192 `#1413` family
- canonical result: `1.08128837` score-first TTT BPB (5-seed mean, `sigma = 0.00058943`)
- best measured single-seed follow-up: `R1_e_baseline = 1.08078562`
- important caveat: `R1_e_baseline` ran at `605s`, so it is not a clean lead submission number
- primary contribution: a non-record evidence package documenting A/B/C/D/E, R1-R9, the fixed-Brotli rate-distortion finding, and 12+ negative results

## What This Package Claims

- `D` is the canonical, best-supported base from this branch because it is backed by a clean 5-seed RunPod bundle.
- `R1_e_baseline` is a real measured single-seed follow-up signal on top of `D`, but only as a follow-up signal.
- OWC/CDQuant create a fixed-Brotli compression-entropy penalty on this stack that overwhelms their raw BPB gain under the 16 MB cap.
- The negative-results inventory is substantial enough to be useful to nearby Track A efforts.
- Pegasus validation attempts are operational context only. The main evidence claim in this package is already anchored to the RunPod `D` 5-seed bundle.

## What This Package Does Not Claim

- It does not claim a record or a submission-valid lead result.
- It does not claim multi-seed confirmation for `R1_e_baseline`.
- It does not claim Pegasus produced completion-valid validation evidence.
- It does not claim grouped OWC, CDQuant salvage, or other follow-up ideas are already demonstrated on this stack.
- It does not treat missing Pegasus reruns as a gap in the main evidence line.

## Primary Documents

- Longform report: [REPORT.md](REPORT.md)
- Artifact inventory: [ARTIFACT_MAP.md](ARTIFACT_MAP.md)
- Submission metadata: [submission.json](submission.json)

## Included Files

- `README.md`
- `REPORT.md`
- `submission.json`
- `requirements.txt`
- packaged canonical `train_gpt.py`
- canonical `D` train logs:
  `train_seed0.log`, `train_seed42.log`, `train_seed1234.log`, `train_seed1337.log`, `train_seed2025.log`
- machine-readable summaries:
  `d_submission_summary.tsv`, `r_series_combined_summary.tsv`
- best follow-up eval log:
  `r1_e_baseline.log`
- `ARTIFACT_MAP.md`

## Canonical Packaged Script

- packaged script: [train_gpt.py](train_gpt.py)
- SHA256:
  `4f2ab2ca43105e94ea1b09924a7580a5446c72be47c2ff1d580c9c604fba69dd`
- archived source paths used to create the package-local script:
  `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/train_gpt.py`
  `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/ngram_tilt.py`
  `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/fused_expert_kernel.cpp`
- identity note:
  the package-local `train_gpt.py` is a single-file consolidation of the archived seed-0 `train_gpt.py` plus its archived helper chain; this preserves the submission-shaped requirement that counted code live in `train_gpt.py`

## Runtime Notes

- A minimal dependency list is provided in [requirements.txt](requirements.txt).
- The package-local `train_gpt.py` inlines the archived n-gram helper chain instead of shipping separate local code files.
- The measured runs used the official Parameter Golf / RunPod CUDA environment.
- The packaged script expects a FlashAttention runtime exposing `flash_attn_interface`; use the challenge image or an equivalent Hopper-compatible install.
- The inlined n-gram helper writes `fused_expert_kernel.cpp` to the records folder at runtime if it is absent, then compiles `libfused_ngram.so` with `g++`.

## Reviewer Snapshot

- [x] Canonical `D` evidence is packaged with train logs for the five reported seeds
- [x] `R1_e_baseline` evidence is packaged only as a follow-up eval log, not as the canonical submission basis
- [x] The package-local `train_gpt.py` consolidates the archived seed-0 script and helper chain into a single counted code file
- [x] The package-local `train_gpt.py` is checksum-verified as the packaged submission-shaped review artifact
- [x] Non-record framing remains explicit; this folder does not claim a clean lead result or a grouped-OWC success

## Provenance Note

Do not treat the mutable working-tree file at
`records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/train_gpt.py`
as the canonical measured artifact for this package.

That path is a mutable prep surface. The provenance anchors for the packaged script are the archived seed-0 source paths listed above, while the package-local `train_gpt.py` is the self-contained single-file review artifact derived from them.
