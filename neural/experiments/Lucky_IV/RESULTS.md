# Lucky IV Results

## Configuration
- Base: SLOT_brotli (Rascal II safepoint + brotli + SLOT)
- Change: SLOT_STEPS 8 → 24 (single variable, eval-only)
- Delta: shared (1,1,dim) — NOT per-sample
- Compression: brotli-11 + byte-shuffle (stride=2)
- SLOT_ENABLED=1, SLOT_STEPS=24
- SKIP_GPTQ=1 (naive int6, GPTQ calibrates 2 layers)
- loader: coprime, cache:4, shards_per_batch:1
- Hardware: 8xH100 SXM, 600s wallclock cap
- File: `neural/experiments/Lucky_IV/train_gpt.py`

## Seed 300

| metric | value |
|---|---|
| steps | 6296 |
| step_avg | 90.55ms |
| post_ema_bpb | 1.1355 |
| model int6+brotli | 15,400,401 bytes |
| total submission | 15,524,362 bytes |
| code size | 123,961 bytes |
| roundtrip_bpb | 1.14522787 |
| **sliding+slot24_bpb** | **1.09600210** |
| eval_time | 492,743ms (~8.2 min) |

## Seed 444 — PENDING

## Seed 42 — PENDING

## Comparison vs Slot Machine (SLOT_brotli, slot8)

| metric | Lucky IV (s300) | Slot Machine (s300) | delta |
|---|---|---|---|
| sliding BPB | 1.09600210 | 1.10448947 | **-0.00849** |
| model bytes | 15,400,401 | 15,408,618 | -8,217 |
| total bytes | 15,524,362 | 15,532,578 | -8,216 |
| eval time | 492,743ms | 304,615ms | +188,128ms |

## Comparison vs Rascal II Leader (1.10986874)

| metric | Lucky IV (s300) | Leader (s300) | delta |
|---|---|---|---|
| sliding BPB | 1.09600210 | 1.10979099 | **-0.01379** |

## What Changed vs Slot Machine
1. SLOT_STEPS: 8 → 24 (single variable)
2. Everything else identical

## Key Finding
- SLOT 24 steps worth ~0.0085 BPB over SLOT 8 steps
- Per-sample delta (bsz,1,dim) is HARMFUL — confirmed in earlier Lucky IV run (1.1234 BPB)
- Shared delta (1,1,dim) is correct — cross-batch regularization matters
- Eval time 492s fits within 600s eval cap

## Promotion Gate
Beat 1.10986874 on seed 444 → confirm on seed 300 → update LEADER.md

Seed 300 result: **1.09600210 PASSES** (beats 1.10986874 by 0.01387)
Seed 444: PENDING
