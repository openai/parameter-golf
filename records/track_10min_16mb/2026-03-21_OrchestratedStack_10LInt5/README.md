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
- **Optional LeakyReLU² MLP** (`LEAKY_RELU_SLOPE`, default `0`): set e.g. `LEAKY_RELU_SLOPE=0.5` for `F.leaky_relu(..., negative_slope=0.5).square()` instead of `relu²`, aligned with the top record line (see `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`). **A/B this vs baseline on the same seed** for a distinct submission story (`docs/PLAN-leaderboard-novel-improve.md`).
- Module docstring + removed stray end-of-file comments.

## Dependencies

**Important:** `requirements.txt` is only at the **repository root**. **`requirements-record.txt` is only in this folder**, not in the root — so `pip install -r requirements-record.txt` must be run **from this directory**, or use the **full path** from the root (see below).

From **repository root** (folder that contains `requirements.txt`):

```bash
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
pip install -r records/track_10min_16mb/2026-03-21_OrchestratedStack_10LInt5/requirements-record.txt
```

Or a single root file (same end result):

```bash
pip install -r requirements-orchestrated-stack.txt
```

From **this** record directory only:

```bash
cd records/track_10min_16mb/2026-03-21_OrchestratedStack_10LInt5
pip install -r requirements-record.txt   # zstd only; install repo root requirements.txt first
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
# Optional ablation (top-SOTA family uses 0.5):
# export LEAKY_RELU_SLOPE=0.5

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-21_OrchestratedStack_10LInt5/train_gpt.py
```

Or use [`scripts/run_orchestrated_stack_8xh100.sh`](../../../scripts/run_orchestrated_stack_8xh100.sh) from the repository root (`bash scripts/run_orchestrated_stack_8xh100.sh`).

## Budget smoke (~$25 workflow)

For **NCCL / compile / OOM** checks without paying for full sliding eval + export, use **`SMOKE_MODE=1`** (skips validation during training and skips int export + final eval). **Do not** report `val_bpb` from a smoke run as a leaderboard number.

```bash
bash scripts/smoke_orchestrated_8xh100.sh
# optional: LEAKY_RELU_SLOPE=0.5 bash scripts/smoke_orchestrated_8xh100.sh
```

Full **A/B** (baseline vs LeakyReLU²) with production protocol:

```bash
bash scripts/run_orchestrated_full_ab.sh baseline
bash scripts/run_orchestrated_full_ab.sh leaky
```

Schedule and env rationale: [`docs/PLAN-h100-novel-budget.md`](../../../docs/PLAN-h100-novel-budget.md).

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
