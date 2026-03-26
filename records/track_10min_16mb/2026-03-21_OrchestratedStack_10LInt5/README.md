# Orchestrated stack — 10L Int5 MLP + SmearGate + BigramHash + Muon WD + SWA

**Status:** Code-ready for **real 8×H100** training. Fill in `submission.json` and add `train.log` after your run before any leaderboard PR.

## Gap analysis (root vs competitive stack)

| Area | Repo root `train_gpt.py` | This record (aligned with SOTA `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`) |
|------|---------------------------|----------------------------------------------------------------------|
| Depth / MLP | 9L, MLP 2× | **10L**, **MLP 3×** (hidden 1536) |
| Quantization | int8 + zlib roundtrip | **Mixed int5 (MLP) / int6 (attn)**, **zstd-22** |
| Embedding extras | — | **SmearGate**, **BigramHash(10240)** |
| Optimization | Basic Muon | **Muon + WD 0.04**, momentum warmup, **SWA** (late training) |
| Eval | Chunked val | **Sliding window** `EVAL_STRIDE=64` (final metric) |
| Bytes budget | Baseline ~15.8MB class | Tuned for **<16MB** decimal total with zstd |

**Further ideas (not implemented here — for future experiments):** merge **int6 QAT (STE)** from 11L records, push **BigramHash** bucket count with strict byte accounting, doc-boundary eval ablations (see LoRA TTT record).

## Changes vs `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`

- **Default `EVAL_BATCH_SEQS=64`** (was 32): same **val_bpb** math, faster sliding eval (helps stay under **eval time** cap).
- **Startup warning** if `zstandard` is missing (zlib fallback inflates artifact size).
- Module docstring + removed stray end-of-file comments.

## Dependencies

```bash
source .venv/bin/activate
uv pip install -r requirements-record.txt
```

## Run (8×H100 SXM, ~600s training cap)

From repo root (or container `/workspace/parameter-golf`), with FineWeb sp1024 data and tokenizer paths set:

```bash
export NCCL_IB_DISABLE=1
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export RUN_ID=orchestrated_10l_int5
export MAX_WALLCLOCK_SECONDS=600
export SEED=42

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_OrchestratedStack_10LInt5/train_gpt.py
```

Or use [`scripts/run_orchestrated_stack_8xh100.sh`](../../../scripts/run_orchestrated_stack_8xh100.sh) from the repository root (`bash scripts/run_orchestrated_stack_8xh100.sh`).

## Submission checklist (official README)

After training:

1. Copy the printed **`final_int8_zlib_roundtrip_exact`** lines into this README and `submission.json`.
2. Save full stdout/stderr as **`train.log`**.
3. Confirm **bytes_total** ≤ 16,000,000 and training wallclock ≤ 600s on 8×H100 SXM for record track.
4. For SOTA claims: **≥0.005 nats** vs current best with **p < 0.01** across seeds (see root README).

## Files

- `train_gpt.py` — trainer snapshot  
- `requirements-record.txt` — `zstandard`  
- `submission.json` — **placeholder** until you run  

---

*Orchestrated implementation: explorer (records survey), backend-specialist (trainer fork), performance-optimizer (eval batch throughput), documentation-writer (this README).*
