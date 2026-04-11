# Non-Record Submission: Superchunk BPE (vocab 1024)

Short run checking **superchunk-trained Rust BPE** (`tokenizer.pkl` + re-exported `fineweb10B_superchunk1024` shards) against the stock **1024-vocab** training recipe on **8×H100**, **600s** wall clock, **non-record / 16MB** track.

## What superchunking is

Standard GPT-style BPE (here: **rustbpe** + tiktoken) learns merges **inside** regex-defined chunks (words, numbers, etc.). **Superchunk BPE** adds a **second phase**: it builds sequences where each chunk is represented as a **single phase-1 token**, then learns **cross-chunk** merges; those merges are **interleaved by frequency** with phase-1 merges into **one** merge table. At inference there is no separate “superchunk mode”—behavior is whatever that combined table encodes.

## Data and setup

- **Tokenizer:** superchunk BPE, **vocab 1024** (same width as SP1024 baseline family).
- **Shards:** `fineweb10B_superchunk1024` from `export_fineweb_custom_bins.py` on `docs_selected.jsonl`.
- **Training:** `train_gpt.py` with `TOKENIZER_PATH` pointing at the Rust BPE directory (`tokenizer_kind=rust_bpe` in log).

## Results (from `train.log` tail + `submission.json`)

| Metric | Value |
|--------|--------|
| **Steps** (wall stop) | **9,131** / 20,000 (`stopping_early: wallclock_cap`) |
| **Wall time** | ~**600 s** |
| **Pre-quant `val_bpb`** (last eval, step 9131) | **1.2294** |
| **Pre-quant `val_loss`** | **2.0968** |
| **Post–int8+zlib round-trip `val_bpb`** | **1.23893525** |
| **Post–int8+zlib `val_loss`** | **2.11308352** |
| **`bytes_total` (int8+zlib + code)** | **15,868,556** (~15.1 MiB) |
| **`bytes_model_int8_zlib`** | 15,818,828 |
| **Peak GPU memory** | ~**10.2 GiB** / rank (log) |
| **Model params** | **17,059,912** |

Validation checkpoints in the log show `val_bpb` trending down through the run (e.g. **1.3844** @ step 1000 → **1.2294** @ step 9131).

## Included files

- `train_gpy.py` — training script snapshot for this run (filename as stored).
- `train.log` — full stdout (includes pasted source + per-step metrics).
- `submission.json` — leaderboard-style metadata for this entry.
