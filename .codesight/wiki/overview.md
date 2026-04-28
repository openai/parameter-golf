# parameter-golf — Overview

> **Navigation aid.** This article shows WHERE things live (routes, models, files). Read actual source files before implementing new features or making changes.

**parameter-golf** is a python project built with raw-http.

## Scale

46 library files · 2 middleware layers · 27 environment variables

**Libraries:** 46 files — see [libraries.md](./libraries.md)

## Required Environment Variables

- `ABLATION_EVAL` — `train_gpt_verbose.py`
- `COMPILE_SHAPE_PADDING` — `_archive/logs/remote_run_20260413/train_gpt_tail.py`
- `COMPILE_TRITON_CUDAGRAPHS` — `_archive/logs/remote_run_20260413/train_gpt_tail.py`
- `DATA_PATH` — `_archive/records/track_10min_16mb/2026-03-25_Ternary_Feedback_TTT/iso_compute_bakeoff.py`
- `DDP_FIND_UNUSED_PARAMETERS` — `_archive/logs/remote_run_20260413/train_gpt_tail.py`
- `DISABLE_TRITON` — `_archive/logs/remote_run_20260413/train_gpt_tail.py`
- `ENGRAM_COMPETITION_ENABLED` — `_archive/logs/remote_run_20260413/train_gpt_tail.py`
- `FAST_SMOKE` — `_archive/logs/remote_run_20260413/train_gpt_tail.py`
- `FAST_SMOKE_BATCHES` — `_archive/logs/remote_run_20260413/train_gpt_tail.py`
- `GRAD_ACCUM_STEPS` — `_archive/logs/remote_run_20260413/train_gpt_tail.py`
- `LOCAL_RANK` — `_archive/logs/remote_run_20260413/train_gpt_tail.py`
- `MATCHED_FINEWEB_REMOTE_ROOT_PREFIX` — `data/cached_challenge_fineweb.py`
- _...15 more_

---
_Back to [index.md](./index.md) · Generated 2026-04-15_