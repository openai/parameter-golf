# Vault — locked source files

## train_gpt_rascal_sota_REAL.py
- sha256: 0ec1f462ab39fd601b18f2b086f6283a0c8db3d2a9780a92dfb206ec46e067cb
- git source: 946f0a7:experiments/SOTA/2026-03-30_JUNKYARD_RAT_RASCAL_II_nogptq/train_gpt.py
- code bytes: 118521 (matches seed444 log exactly)
- result: 1.10986874 BPB (seed 444), 3-seed mean 1.1099
- run with: SKIP_GPTQ=1 (file has GPTQ code but flag skips it)
- DO NOT REPLACE. If this file changes, re-run before any claim.

## What is NOT the real file
- records/track_10min_16mb/2026-03-30_Rascal_8xH100/train_gpt.py
  hash 7b5bffe, 103437 bytes — stripped version, was never run
- analysis/pr1120_racecar_lab/copies/train_gpt_rascal_sota_local.py
  hash b83da176, 121545 bytes — different again, never run
