# Artifact Map

This folder is designed to stand on its own as a non-record records package.

## Packaged Review Artifacts

- report: [REPORT.md](REPORT.md)
- overview: [README.md](README.md)
- metadata: [submission.json](submission.json)
- runtime manifest: [requirements.txt](requirements.txt)
- packaged canonical script: [train_gpt.py](train_gpt.py)

## Packaged Canonical D Evidence

- machine-readable canonical summary:
  [d_submission_summary.tsv](d_submission_summary.tsv)
- canonical train logs:
  [train_seed0.log](train_seed0.log)
  [train_seed42.log](train_seed42.log)
  [train_seed1234.log](train_seed1234.log)
  [train_seed1337.log](train_seed1337.log)
  [train_seed2025.log](train_seed2025.log)

## Packaged R-Series Evidence

- best measured single-seed follow-up eval log:
  [r1_e_baseline.log](r1_e_baseline.log)
- machine-readable R-series summary:
  [r_series_combined_summary.tsv](r_series_combined_summary.tsv)

## Script Provenance

- packaged script SHA256:
  `4f2ab2ca43105e94ea1b09924a7580a5446c72be47c2ff1d580c9c604fba69dd`
- package-local script role:
  single-file consolidation of the archived seed-0 `train_gpt.py` plus its archived helper chain, produced to keep counted code inside `train_gpt.py`
- archived source paths used to create the package-local script:
  `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/train_gpt.py`
  `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/ngram_tilt.py`
  `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed0/pr1413_combo_s0/fused_expert_kernel.cpp`
- archived source component SHA256s:
  - `train_gpt.py`: `db19d2a078354bd861e425965badbdb41ad644a2aec9c1c9a4f6984fca4c7019`
  - `ngram_tilt.py`: `065ced48efcd5ae633f4307d254a0d3e475641878a0dc580f8e677b6e56aa379`
  - `fused_expert_kernel.cpp`: `6b11646609508a84f7c2d9ddd9cdb4c133c2474ec83a50b78313d96664984056`
- additional archived canonical `D` script paths:
  - `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed42/pr1413_combo_s42/train_gpt.py`
  - `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed1234/pr1413_combo_s1234/train_gpt.py`
  - `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed1337/pr1413_combo_s1337/train_gpt.py`
  - `artifacts/runpod_pull/pr1413_archive_20260407_213205/seed2025/pr1413_combo_s2025/train_gpt.py`

## External Archive Roots Used To Build This Package

These paths are provenance references and are not required to review the packaged folder itself.

- canonical `D` archive root:
  `artifacts/runpod_pull/pr1413_archive_20260407_213205/`
- R-series archive root:
  `artifacts/runpod_pull/runpod_r_experiments_20260409_182045/`

## Important Note

Do not treat the mutable working-tree file at
`records/track_10min_16mb/2026-04-07_SP8192_QK5_LegalTTT_ParallelResid7_TiltPrep/train_gpt.py`
as the canonical measured artifact for this package.

That path is a mutable prep surface. The packaged `train_gpt.py` is the self-contained review copy, and the archived seed-0 path above is the provenance anchor for that copy.
