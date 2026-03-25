# Non-record dev snapshot (1×H100)

**Submission type:** **Non-record** work-in-progress — **not** a record-track or public-leaderboard SOTA claim.

This folder is a **paired** bundle: the `train.log` and `train_gpt.py` here are from the **same** autoresearch run (see `submission.json` metrics). Use it for grant linkage / reproducibility; it reflects **current** harness work, not an older experiment name.

## What this run is

- **Hardware:** 1×GPU, `torchrun --nproc_per_node=1`, autoresearch **`baseline-1gpu`** preset, **600s** wallclock cap.
- **Stack (from `train.log`):** `FAKE_QUANT_INT8` with late-window STE on `CastedLinear`, **`torch.compile` with `fullgraph=False`**, int8 export clip percentile **99.995**, seed **12345**, GQA attention.
- **Not** the official **8×H100 / 10 min** record configuration.

## Metrics (from `train.log`; deciding line per harness)

- **`final_int8_zlib_roundtrip_exact val_bpb`:** **1.35151831** (lower better)
- **`val_loss`:** 2.28198205
- **Total submission size int8+zlib:** 12,622,882 bytes (under 16,000,000)
- **Stopped:** wallclock cap at step **1004 / 20000**
- **Peak memory:** ~16,438 MiB allocated

## Historical note

An **earlier** harness row in `results.tsv` reached **~1.33** `val_bpb` on a different recipe (`070-equivalent SEED12345`); the **training log for that run is not paired here** (overwritten by later jobs). This PR bundle intentionally shows the **best run for which we still have a full local log + script**.

## Reproduce

From a CUDA machine with challenge **FineWeb SP-1024** data and tokenizer on disk, run **`train_gpt.py` from this folder**. Paths in `train.log` header (`wrapper_workdir`, dataset locations) show the layout used for the logged run—adjust to your tree.

## Files (only these four)

| File | Purpose |
|------|---------|
| `train_gpt.py` | Exact script for this run |
| `train.log` | Full stdout |
| `submission.json` | Metadata (`github_id`: **jaksenc**) |
| `README.md` | This description |

## Not included

`autoresearch/`, `data/`, checkpoints — per standard non-record hygiene.
