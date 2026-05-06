# Run Log

## How to read this
- **val_bpb**: lower is better. Current SOTA is 1.1748.
- **iterations**: full run is ~20,000. Smoke tests use 100–500.
- **Status**: smoke | full

---

## Runs

| Date | File | Key Config | val_bpb | Iterations | Status | Notes |
|------|------|-----------|---------|------------|--------|-------|
| 2026-03-21 | train_gpt_rank1_int5mlp_swa.py | TRIGRAM_HASH_BUCKETS=4096 | 2.1541 | 100 | smoke | trigram works, high bpb expected at 100 iters |
| 2026-03-21 | train_gpt_recurrent.py | NUM_LOOPS=1 NUM_LAYERS=10 | 3.1511 | 100 | smoke | baseline parity — model_params=25517137, step_avg=696ms, submission_size=16.49MB (slightly over 16MB limit, needs attention) |
| | train_gpt_recurrent.py | NUM_LOOPS=3 NUM_LAYERS=4 | TBD | 100 | planned | 3-loop recurrence smoke test — compare model_params and step_avg vs baseline |

---

## Ideas Queue

| Idea | Risk | Potential | Status |
|------|------|-----------|--------|
| Depth recurrence (NUM_LOOPS=3, NUM_LAYERS=4) | medium | high — no one has submitted this | in progress |
| Trigram hash embedding | low | low-medium — incremental over bigram | implemented |
| Tetragram hash | low | low | not started |
| Wider dims with freed recurrence budget | low | medium | blocked on recurrence results |
| Int4 MLP with QAT | high | high | not started |
