# Parameter Golf — Roadmap to SOTA

**Current best:** 1.2590 bpb (2×H100, SP1024)
**Target:** Beat official baseline (~1.20) on 8×H100 → submit to non-record track
**Stretch target:** Approach SOTA (1.08) with as many techniques as we can implement
**Deadline:** April 30, 2026 (12 days remaining)

---

## Phase 1: Easy Wins (env vars + small code changes)

No new infrastructure needed. Each can be tested in one 10-min run.
Estimated total improvement: ~0.01-0.02 bpb

| # | Change | Type | Effort | Expected Impact |
|---|--------|------|--------|-----------------|
| 1 | LeakyReLU(0.5)² | 1 line code | 5 min | +0.002-0.005 |
| 2 | Weight decay 0.09 | env var | 0 min | +0.002-0.004 |
| 3 | Warmdown 72% of training | env var | 0 min | +0.001-0.003 |
| 4 | Recurrence start mid-training | ~5 lines code | 15 min | +0.002-0.005 |
| 5 | EMA weights (decay=0.997) | ~20 lines code | 30 min | +0.002-0.005 |
| 6 | Parallel residuals (layer 7+) | ~15 lines code | 30 min | +0.002-0.004 |

**Implementation order:** 1 → 2 → 3 (test together as one run), then 4, then 5+6.

---

## Phase 2: Medium Effort (architecture + quantization)

Needs code changes and testing. ~1-2 hours implementation each.
Estimated total improvement: ~0.02-0.04 bpb

| # | Change | Type | Effort | Expected Impact |
|---|--------|------|--------|-----------------|
| 7 | 11 layers + mlp_mult=4 | env vars + budget check | 30 min | +0.01-0.02 |
| 8 | Partial RoPE (16/64 dims) | ~10 lines code | 30 min | +0.002-0.005 |
| 9 | MuonEq-R optimizer | ~30 lines code | 1 hr | +0.005-0.01 |
| 10 | Brotli-11 compression | replace zlib call | 15 min | +0.001-0.002 (indirect, more room) |

**Note on #7:** 11 layers + mlp_mult=4 = ~45M params. With INT8 that's 45 MB — way over 16 MB.
This REQUIRES INT6 quantization (#11) to fit. So #7 and #11 must go together.

---

## Phase 3: Hard (new infrastructure)

Needs significant work or external tools. ~2-4 hours each.
Estimated total improvement: ~0.05-0.10 bpb

| # | Change | Type | Effort | Expected Impact |
|---|--------|------|--------|-----------------|
| 11 | GPTQ INT6 quantization | ~100 lines code | 3-4 hrs | +0.02-0.04 (enables larger model) |
| 12 | SP8192 tokenizer | data pipeline | 2-3 hrs | +0.03-0.05 (biggest single lever) |
| 13 | Legal TTT | ~50 lines code | 2 hrs | +0.002-0.003 |

---

## Proposed Schedule (12 days remaining)

### Day 1 (today): Phase 1 easy wins
- Implement LeakyReLU, weight decay, warmdown in one combined run
- Implement staged recurrence (RECUR_START_STEP)
- Run 3-4 experiments on 2×H100 (~$6)

### Days 2-3: Phase 1 continued + Phase 2 start
- Implement EMA weights
- Implement parallel residuals
- Implement partial RoPE
- Run 4-5 experiments (~$8)

### Days 4-6: Phase 2 + Phase 3 start
- Implement MuonEq-R optimizer
- Start GPTQ INT6 quantization
- Start SP8192 tokenizer pipeline
- Run experiments as each piece is ready

### Days 7-9: Phase 3 + integration
- Complete GPTQ + SP8192
- Combine all working techniques
- Run combined config on 2×H100
- Tune hyperparameters

### Days 10-11: Final runs
- Rent 8×H100 for official benchmarks (~$3.30/run × 3 seeds = $10)
- Run best config with 3 seeds for statistical significance
- Prepare submission folder

### Day 12 (April 30): Submit
- PR to parameter-golf repo
- Non-record track submission

---

## Budget Estimate

```
Phase 1 experiments: ~$6
Phase 2 experiments: ~$10
Phase 3 experiments: ~$15
Final 8×H100 runs:  ~$15
Total:              ~$46
Current balance:    ~$54 (should be enough)
```

---

## Success Criteria

| Level | val_bpb (8×H100) | What it means |
|-------|-------------------|---------------|
| Minimum | < 1.20 | Beat official baseline → valid non-record submission |
| Good | < 1.15 | Competitive with early leaderboard entries |
| Great | < 1.10 | Approaching SOTA territory |
| Amazing | < 1.08 | Matching or beating current SOTA |

Our 2×H100 result of 1.259 extrapolates to roughly ~1.18 on 8×H100, which already beats the minimum target. With Phase 1+2 improvements we should land in the "Good" range (~1.12-1.15).
