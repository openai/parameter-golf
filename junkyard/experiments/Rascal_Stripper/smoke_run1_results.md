# Smoke Run 1 — 2026-03-30

**Status:** All 4 variants are identical (safe copy — no changes applied yet)
**Purpose:** Establish baseline + verify smoke test infrastructure

## Final BPB (final_sliding_window_exact, stride=64)

| Variant     | val_bpb      | delta vs baseline |
|-------------|--------------|-------------------|
| baseline    | 1.22946945   | —                 |
| turbomuon   | 1.22928182   | -0.00019          |
| engramlite  | 1.22908999   | -0.00038          |
| combo       | 1.22919608   | -0.00027          |

**Note:** All 4 scripts are identical — deltas are GPU noise (~±0.0004), not signal.

## Val BPB Curve (baseline only, representative)

| Step | val_bpb |
|------|---------|
|    0 | 4.1049  |
|  300 | 1.4775  |
|  600 | 1.3620  |
|  900 | 1.3176  |
| 1200 | 1.2821  |
| 1500 | 1.2278  |

## Infrastructure Notes

- Smoke test runs correctly: 4 × 1500 steps sequential, ~128s per variant
- Step time: ~85.8ms/step on 8×H100
- **BUG:** results table showed N/A — grep looked for `final_sliding_window_s64_exact`
  but log writes `final_sliding_window_exact` (s64 suffix only appears on secondary eval).
  Fixed in smoke_test.sh.

## Next Steps

1. Implement TurboMuon in `train_gpt_turbomuon.py`
2. Implement EngramLite in `train_gpt_engramlite.py`
3. Re-run smoke — look for >0.002 BPB improvement to call it signal over noise
