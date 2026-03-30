# Family 1A — tied blocks (non-record, 1×H100 dev)

**Submission type:** **Non-record** work-in-progress. This is **not** a leaderboard/SOTA entry: the public 10-minute **8×H100** baseline is stronger on `val_bpb`. Goal here is a **reproducible snapshot** of **Family 1 / Batch 1A** (tied block weights + stable training recipe) from the autoresearch harness, suitable for a **draft PR** (e.g. compute-grant link) without claiming a record.

## Before you open the upstream PR

1. **`submission.json`** already has **`github_id`:** `jaksenc`. Optionally edit **`author`** if you want your display name on leaderboard metadata (default: `Challenge participant`).
2. Keep this folder to **only** the four files below—do **not** add checkpoints, datasets, or `autoresearch/`.

## What changed (high level)

- **Tied transformer block weights** across layers (`TIE_BLOCK_WEIGHTS`), with per-layer norms / gates unchanged (Family 1A).
- **Training recipe:** global **grad clip 1.0**, **30-step** linear data warmup, Muon + AdamW defaults as in `train_gpt.py`; **600s** wallclock cap.
- **Hardware:** **1×GPU** (`torchrun --nproc_per_node=1`), autoresearch **`baseline-1gpu`** preset — **not** the official record track (8×H100 / 10 minutes).

## Metrics (from `train.log`)

- **Final metric:** `final_int8_zlib_roundtrip_exact val_loss:2.55636938 val_bpb:1.51402594`
- **Total submission size (int8+zlib):** 2,033,640 bytes (under 16,000,000)
- **Code size:** 51,784 bytes
- **Stopped:** wallclock cap at step **1258 / 20000**
- **Peak memory:** ~9989 MiB allocated

**Note:** The same codebase + recipe produced **~1.507** `val_bpb` on another seed/run (`results.tsv` / `FAMILY_1A_LEARNINGS.md`); this log is the **paired** artifact for the **current** `train_gpt.py` snapshot.

## Reproduce (representative)

From a CUDA machine with the challenge **FineWeb SP-1024** data and tokenizer on disk, run **`train_gpt.py` from this folder** (or copy into a clean clone root and point paths). Example shape:

```bash
RUN_ID=family1a_nonrecord \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Adjust paths to match your layout. Full wrapper context is in **`train.log`** header (`wrapper_command`, `wrapper_workdir`, shard layout).

## Included files (only these)

| File | Purpose |
|------|--------|
| `train_gpt.py` | Exact training script for this log |
| `train.log` | Full stdout for the run |
| `submission.json` | Challenge metadata (`github_id`: **jaksenc**; optional **`author`**) |
| `README.md` | This description |

## Intentionally **not** included

- **`autoresearch/`** harness (local iteration only; upstream repo does not require it for `records/`).
- **Checkpoints** (`.pt`, `.ptz`, etc.) — not part of the standard record bundle.
- **`data/`** — download via upstream `data/cached_challenge_fineweb.py`.
- **Secrets** — none present; do not add API keys or `.env`.

## Promotion provenance

Promoted from `autoresearch/dev/current_submission` on **2026-03-23** (`promote_record.py`). No existing `records/` entries were modified or removed.
