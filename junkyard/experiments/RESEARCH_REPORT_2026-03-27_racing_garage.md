# Parameter Golf Racing Garage Report

**Date:** 2026-03-27  
**Scope:** Issue #140 starting point, latest buzz/techniques, legality risk, and immediate wins for a non-ngram-leaning team.

## Executive Summary

1. **Official record is still 1.1194 BPB** (`#549`, merged 2026-03-24), but open PR claims now span down to `0.0xx` and even `0.00000035` via eval-time cache systems.
2. **The low-BPB cache frontier is under active legitimacy challenge**: maintainers explicitly said on 2026-03-27 they are investigating hashing/renormalization issues and considering restrictions or a separate leaderboard (`#677`).
3. **Your non-ngram position is genuinely strong**: local Rat Rod v1 reports `1.1129` sliding base with `6882` steps in 600s, beating current official record quality without relying on contested cache mechanics.
4. **Highest-confidence wins now** are likely in legal neural tuning (TTT recipe quality, quantization gap reduction, throughput/stability discipline), not in two-pass/full-rescore cache escalation.
5. **Strategic call:** run two lanes in parallel:
   - **Lane A (primary):** compliance-first neural/non-ngram improvement and submission hardening.
   - **Lane B (hedge):** only probability-valid, causality-safe cache experiments with explicit normalization checks.

## Competition State (as of 2026-03-27)

### 1) Official vs Pending Split

- **Official merged leaderboard top:** `1.1194` (`#549`, LeakyReLU^2 + legal score-first TTT + Parallel Muon).
- **Pending/open claims:**
  - `#944`: `0.0165` (packed causal memory + Dirichlet mixing)
  - `#962`: `0.0214` (packed training n-gram artifact + learned gate)
  - `#945`: `0.0274` (order-16 frozen n-gram oracle + learned gate + TTT)
  - `#931`: `0.0498`
  - `#913`: `0.0887` with a tiny model and cache-heavy eval
  - `#959`: `0.00000035` (submitted partly to demonstrate rule boundary concerns)

### 2) Rule and Measurement Uncertainty Is Real

From `#677` and linked PR discussions:

- Maintainers are **not merging** this wave yet while reviewing correctness.
- Core challenge: many methods mix probabilities only for the correct token and may not form a proper distribution over all vocab tokens.
- Two-pass/full-rescore causality is heavily contested.
- Possible future outcomes discussed publicly:
  - stricter normalization requirements,
  - explicit causality rule tightening,
  - separate leaderboards (adaptive eval cache vs strict predictor track),
  - eval-time resource limits.

## Latest Buzz Techniques (Triaged)

### Cache-Dominant Wave (very active, high rule volatility)

- **Packed training n-gram memory inside artifact** (`#931`, `#944`, `#962`)
  - Strong claimed gains.
  - Better compliance posture than ad-hoc runtime cache prefill, but still exposed to normalization scrutiny.
- **Two-pass/full-rescore cache pipelines** (`#870`, `#893`, `#933`, `#959`)
  - Massive claimed gains.
  - Highest causality/legality risk right now.
- **Dirichlet/Bayesian mixing and per-order concentration** (`#900`, `#944`)
  - Most technically defensible cache-mixing direction if cache lane survives.
- **Log-space calibration / log-odds blending** (seen in `#933`, `#959`, inspired by Nacrith-style ideas)
  - Promising engineering trick, but currently entangled with controversial eval procedures.

### Neural-First Wave (lower rule risk, harder but durable)

- **Improved legal TTT recipes** (`#953`): per-layer LR groups + cosine schedule.
  - High practical relevance to your stack.
- **Pure-neural architecture claims** (`#875` GDN at `1.0226`)
  - High upside if reproducible; currently limited external validation signal.
- **Hybrid recurrence/SSM attempts** (`#852`, `#857`)
  - Mixed quality; some reproducibility concerns raised in-thread.
- **MoD / MTP / sequence curriculum exploratory PRs** (`#955`, `#956`, `#957`)
  - Low-cost to screen, unknown upside.

### Paper Chatter vs Reality Check

- **Orthogonal residual updates** were explicitly tested in-thread and reported as a regression in that setup.
- **ARO optimizer** is discussed but not yet clearly established as a 600s-budget winner in competition conditions.
- **Nacrith-inspired components** are entering PRs, but mostly inside high-volatility cache-heavy eval stacks.

## What Matters for Your Garage (Non-ngram Advantage)

### Proven Local Strength

From your local docs/logs:

- Rat Rod v1 base: `1.1129` sliding, `6882` steps, `87.20ms/step` (600s budget).
- You already identified and killed weak ideas quickly (Synapse variants, Siphon objective, some complementary settings).
- You have a working H100-first harness mindset (Cobra), focused on base signal isolation and throughput floors.

### Current Internal Risks/Blockers

- Cobra remote A/B logs show **hard failures**, not valid outcomes:
  - shard size mismatch on `fineweb10B_sp1024_mini` validation file
  - compile/inductor failures in some remote runs
- This blocks high-quality fast iteration unless fixed first.

## Technique Risk Matrix (Actionable)

| Technique family | Upside (claimed) | Rule risk | Engineering risk | Recommendation |
|---|---:|---:|---:|---|
| Two-pass full-rescore n-gram | Very high | **Very high** | Medium | Avoid as primary lane right now |
| Hash-ratio cache blending without full-vocab normalization checks | Very high (claimed) | **Very high** | Medium | Avoid unless provably normalized |
| Dirichlet posterior mixing with explicit probability validity checks | High | Medium | Medium-high | Keep as hedge lane only |
| Legal score-first TTT with better schedules/per-layer LR | Medium-high | Low-medium | Medium | **Primary near-term win lane** |
| Pure neural architecture changes (e.g., GDN/Delta-style blocks) | Unknown-high | Low | High | Fast-screen in cheap proxy, then escalate |
| Shared-weight recurrence/crawler variants | Low (local evidence) | Low | High | De-prioritize for now |

## 72-Hour Race Plan

### Phase 0 (Immediate, 2-4h): Unblock Reliable Experimentation

1. Fix remote data path integrity (eliminate mini-shard mismatch).
2. Lock one known-good runtime profile for Cobra/Rat Rod (compile mode + Torch version + FA path).
3. Re-run anchor sanity (`seed=1337`) and confirm metric extraction works end-to-end.

**Gate:** no experiments promoted unless run completes with parseable final metrics.

### Phase 1 (Day 1): High-Confidence Neural Gains

1. Start from Rat Rod v1 strong base.
2. Apply compliance-safe TTT recipe improvements inspired by `#953` but keep cache/mixer disabled.
3. Target reducing quantization gap and improving post-EMA -> final path transfer.

**Suggested order:**
- per-layer LR groups in TTT,
- cosine schedule inside TTT,
- controlled freeze-depth sweep,
- SWA cadence and warmdown interaction in full 600s runs.

**Gate to continue:** `>= 0.002` BPB gain vs current no-cache baseline at equal budget.

### Phase 2 (Day 2): High-Upside Neural Exploration

1. Fast-screen a pure-neural alt lane (GDN/Delta-style block concept) in 200s proxy.
2. Promote only if it beats your current proxy anchor with stable throughput.
3. Run one full 600s confirmation on 8xH100.

### Phase 3 (Day 3): Hedge Lane + Submission Hardening

1. If maintainers clarify normalized cache rules, test only probability-valid single-pass methods.
2. Instrument explicit `sum(p_vocab) ~= 1` checks in eval path for every scored token batch.
3. Package a compliance-first submission variant that is resilient to rule tightening.

## Tactical Calls (Bring Wins Back)

1. **Primary bet:** your fastest path to durable leaderboard gains is still neural/base quality + legal TTT refinement.
2. **Do not over-invest** in extreme two-pass cache tricks until rules settle; those gains are large but fragile.
3. **Exploit your edge:** you already have better local non-ngram signal discipline than most teams posting speculative PRs.
4. **Ship with transparency:** always report both contest metric and a conservative reference metric path to reduce merge friction.

## Key Sources

- Issue tracker and live commentary:
  - https://github.com/openai/parameter-golf/issues/140
  - https://github.com/openai/parameter-golf/issues/677
  - https://github.com/openai/parameter-golf/issues/402
- Representative PRs (state as of 2026-03-27):
  - https://github.com/openai/parameter-golf/pull/549
  - https://github.com/openai/parameter-golf/pull/944
  - https://github.com/openai/parameter-golf/pull/962
  - https://github.com/openai/parameter-golf/pull/945
  - https://github.com/openai/parameter-golf/pull/931
  - https://github.com/openai/parameter-golf/pull/913
  - https://github.com/openai/parameter-golf/pull/959
  - https://github.com/openai/parameter-golf/pull/886
  - https://github.com/openai/parameter-golf/pull/953
  - https://github.com/openai/parameter-golf/pull/875
- Local garage data:
  - `experiments/Rat_Rod/PROGRESS.md`
  - `experiments/Cobra/README.md`
  - `experiments/Cobra/HYPOTHESIS.md`
  - `experiments/RESEARCH_REPORT_crawler_signal_analysis.md`
  - `results/vast_cobra_ab/*.log`
