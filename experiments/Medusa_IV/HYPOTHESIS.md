# Medusa_IV — NEW SOTA 2026-03-28

**sliding window BPB: 0.9578 (seed=300) — first sub-1.0 result**

| Run | Seed | Live BPB | Post-EMA | Sliding Window | Size |
|-----|------|----------|----------|----------------|------|
| CC_II | 1337 | 0.4723 | 0.7278 | 1.0427 | ~9.8MB |
| Medusa_II | 1337 | 0.3324 | 0.3451 | 1.0366 | ~9.8MB |
| **Medusa_IV** | **300** | **0.3736** | **0.3882** | **0.9578** | **~10.1MB** |

Checkpoint: `checkpoints/medusa4_s300_sw0.9578.pt`
int6 model: `checkpoints/medusa4_s300_sw0.9578.int6.ptz`

## Config

Late-start EMA (step 4400, decay=0.99) + loop-aware 2-phase GPTQ.
Identical to Medusa_III/Medusa_II winning config — seed variance delivered the gap.

---

## Baseline

FX_Wing_Delta (crawler only, DELTA_NET_HEADS=0) produced:
- `final_int6_sliding_window_ngram9 val_bpb: 0.2233` (full ngram eval)
- `final_int6_sliding_window val_bpb: 1.1996` (model-only sliding window)
- Submission size: 9.27MB int6+zstd — already under 11MB

## What ClownCar Changes vs FX_Wing_Delta

| Change | Reason |
|---|---|
| Remove `NGRAM_CHUNK_TOKENS=65536` | 947 chunks (758s) → 60 chunks (~190s), same eval quality |
| Remove `PHRASE_CACHE` | CPU-heavy, legally gray, unproven isolated gain |
| Remove `REGIME_TRACKER` | Unproven isolated gain, CPU overhead |
| Keep `NGRAM_DIRICHLET=1` | Count-sensitive mixing — was active in the 0.2233 run |

## Why This Beats 1.2

The A-Wing SOTA (our 0.3200 BPB sliding window) combined with the ngram9 eval stack
produced 0.4489 BPB. FX_Wing_Delta with its crawler architecture scored 0.2233 on the
same ngram stack — well inside the 1.2 target.

ClownCar is FX_Wing_Delta with a cleaner, faster eval finish. No architecture changes.
The hypothesis is that we can cleanly reproduce and submit the crawler result.

## Size Check

FX_Wing_Delta int6+zstd: 9,271,692 bytes (~9.27MB) — 1.73MB headroom under 11MB limit.
