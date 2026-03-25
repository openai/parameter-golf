# Experiment Results

Leaderboard #1 to beat: **1.1194 BPB** (must achieve < 1.1144 at p<0.01)

Training limit: 10 min | Eval limit: 10 min (separate)

---

## Experiment 1 — LongContext4096_FullSOTA
`records/track_10min_16mb/2026-03-24_LongContext4096_FullSOTA/`
`ITERATIONS=6000 WARMDOWN_ITERS=1440 EVAL_STRIDE=80`

| Seed | Steps | ms/step | Pre-TTT bpb | Post-TTT bpb | Artifact (bytes) |
|------|-------|---------|-------------|--------------|-----------------|
| 1337 | — | — | — | — | — |
| 42   | — | — | — | — | — |
| 2025 | — | — | — | — | — |
| **Mean** | | | | | |

---

## Experiment 2 — LongContext4096_Int4_16L_FullSOTA
`records/track_10min_16mb/2026-03-24_LongContext4096_Int4_16L_FullSOTA/`
`ITERATIONS=3500 WARMDOWN_ITERS=840 EVAL_STRIDE=80`

| Seed | Steps | ms/step | Pre-TTT bpb | Post-TTT bpb | Artifact (bytes) |
|------|-------|---------|-------------|--------------|-----------------|
| 1337 | — | — | — | — | — |
| 42   | — | — | — | — | — |
| 2025 | — | — | — | — | — |
| **Mean** | | | | | |

---

## Experiment 3 — LongContext4096_Int4_BankQAT (Risky)
`records/track_10min_16mb/2026-03-25_LongContext4096_Int4_BankQAT/`
`ITERATIONS=3500 WARMDOWN_ITERS=840 EVAL_STRIDE=80`

| Seed | Steps | ms/step | Pre-TTT bpb | Post-TTT bpb | Artifact (bytes) |
|------|-------|---------|-------------|--------------|-----------------|
| 1337 | — | — | — | — | — |
| 42   | — | — | — | — | — |
| 2025 | — | — | — | — | — |
| **Mean** | | | | | |

---

## Experiment 4 — LongContext4096_Int6_QAT (Safe)
`records/track_10min_16mb/2026-03-25_LongContext4096_Int6_QAT/`
`ITERATIONS=6000 WARMDOWN_ITERS=1440 EVAL_STRIDE=80`

| Seed | Steps | ms/step | Pre-TTT bpb | Post-TTT bpb | Artifact (bytes) |
|------|-------|---------|-------------|--------------|-----------------|
| 1337 | — | — | — | — | — |
| 42   | — | — | — | — | — |
| 2025 | — | — | — | — | — |
| **Mean** | | | | | |

---

## Summary

| Experiment | Mean Post-TTT BPB | Beat #1? | Artifact |
|------------|------------------|----------|---------|
| 1. LongContext4096_FullSOTA | — | — | — |
| 2. LongContext4096_Int4_16L | — | — | — |
| 3. LongContext4096_Int4_BankQAT | — | — | — |
| 4. LongContext4096_Int6_QAT | — | — | — |
