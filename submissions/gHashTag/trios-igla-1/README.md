# trios-igla-1 — submission metadata

**Parent:** [`../README.md`](../README.md) TRIOS IGLA — First Honest Gate-2 Pass + Research Infrastructure
**Anchor:** `phi^2 + phi^-2 = 3`

## Run-of-record (first honest Gate-2 pass)

| Field | Value |
|---|---|
| canon_name | `fix-verify-s43` |
| account | `acc1` |
| seed | `43` |
| hidden | `1024` |
| lr | inherited from defaults (operator's local `railway up`) |
| ctx | (no `--ctx` flag — fix shipped in [trios-railway@69c3467](https://github.com/gHashTag/trios-railway/commit/69c3467f)) |
| steps | `12000` |
| **final_bpb** | **`1.5492`** |
| final_step | `12000` |
| finished_at | `2026-04-30T18:46:59 UTC` |
| pipeline | post-PR-#56 (`--ctx` accept) + post-PR-#58 (smoke + flush) + post-PR-#59 (panic hook) + post-PR-#61 (byte-disjoint train/val) |
| gate2_threshold | `1.85` |
| **margin** | **`+0.3008`** ✅ |

## BPB trajectory (verbatim from `bpb_samples`)

| step | val_bpb | comment |
|---:|---:|---|
| 1000 | 0.0003 | warmup-artifact zone (trainer printf bug) |
| 2000 | 0.0001 | warmup-artifact |
| 3000 | 0.0000 | warmup-artifact |
| 4000 | 0.0000 | warmup-artifact |
| 5000 | 0.0000 | warmup-artifact |
| 6000 | 0.0000 | warmup-artifact |
| 7000 | 0.0000 | warmup-artifact |
| 8000 | 0.0000 | warmup-artifact |
| 9000 | 7.2781 | post-warmup spike (model "wakes up") |
| 10000 | 1.6935 | converging |
| 11000 | 1.7399 | converging |
| **12000** | **1.5492** | **honest Gate-2 PASS** |

## Files

| File | Purpose |
|---|---|
| `README.md`                  | This file |
| `config.yaml`                | Reproducible config |
| `ledger_2026-04-30.sql.gz`   | Full Neon CSV-in-SQL snapshot (4 tables, 7,570 rows, 186 KB) |

## No model.bin (still)

The post-mortem in [`../CHECKPOINT_POSTMORTEM.md`](../CHECKPOINT_POSTMORTEM.md)
applies. `record_checkpoint()` in the trainer is a stub; Railway storage
is ephemeral; no weight tensors were retrieved before the deadline.

We submit the run **as a verified ledger entry plus the canonical
config**, not as a model artifact. A reviewer can reproduce the run
from the [trainer](https://github.com/gHashTag/trios-trainer-igla) +
[fleet](https://github.com/gHashTag/trios-railway) repos and the config
below; weight retrieval is post-deadline future work
(`CHECKPOINT_POSTMORTEM.md` § Fix path for Gate-3).

phi² + phi⁻² = 3 — R5-honest — first honest Gate-2 pass.
