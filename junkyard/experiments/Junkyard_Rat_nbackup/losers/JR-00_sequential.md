# JR-00 — Sequential Loader Loser

Date: 2026-03-29

## Result

`JR-00` is the clean Rat Rod-style sequential loader baseline for Junkyard_Rat.

It lost the first loader A/B to `JR-01` coprime.

| Variant | Step avg | Post-EMA BPB | Sliding BPB |
|---|---:|---:|---:|
| `JR-00` sequential | `87.08ms` | `1.1354` | `1.11184332` |
| `JR-01` coprime | `91.00ms` | `1.1340` | `1.11056240` |

## Why It Lost

- sequential is faster
- but the coprime walk bought enough quality to overcome the throughput hit
- the sliding gap of about `0.00128` is large enough to keep the winner active

## Re-run

```bash
bash experiments/Junkyard_Rat/losers/run_sequential.sh
```

or:

```bash
SEED=1337 LOADER_MODE=sequential bash experiments/Junkyard_Rat/run.sh
```
