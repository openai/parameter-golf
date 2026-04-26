# Non-record Submission: Online Data Scheduler — `unique_only + lookahead=16` (Signs-of-Life)

**Track:** `track_non_record_16mb` — submitted as a pre-compute-grant **signs-of-life** probe. Numbers below are from **local cheap ablations** (very short proxy runs on a single GPU / MPS), not from a full 8×H100 record run. This PR documents the minimum-viable scheduler policy that we have isolated in local ablations and that we intend to verify at full 8×H100 budget once compute credits are available.

**Author:** SyntagmaNull
**Date:** 2026-04-19
**Status:** signs-of-life / pre-grant — not competing for the leaderboard at this time

---

## TL;DR

We isolate a **minimum-viable online data scheduler policy** — *lookahead buffer + unique-token-ratio selection* — that in local cheap ablations outperforms a `random_lookahead_16` baseline **6/6 across seed × budget** without changing the model, the tokenizer, or the data itself (only the *order* in which raw chunks are fed to the model).

The scheduler path is already wired into a modified `train_gpt.py` on a dedicated scheduler rank (`SCHEDULER_POLICY=score_unique_only`, see `code/train_gpt_scheduler_excerpt.py`). The submitted **real-machine first-experiment** is a single clean contrast:

```
baseline_full_train   vs   7_train_1_scheduler_unique_only_lookahead_16
```

with `6_train_2_scheduler_unique_only_lookahead_16` as an optional third arm if budget allows.

---

## What is the scheduler

Not a new model, not a new tokenizer, not semantic rewriting. The scheduler sits on top of raw FineWeb chunk loading and does only:

1. Maintain a **lookahead buffer** of candidate raw chunks (lookahead size = 16).
2. Score each candidate by **unique-token ratio** (`# unique tokens / # tokens`).
3. Emit the highest-scoring chunk to the training rank; the rest stay in the buffer.

That's the whole policy. The minimum-viable form `score_unique_only + lookahead=16` is deliberately leaner than the earlier `novelty_v1` composite (`unique_ratio + transition_ratio − maxfreq_penalty`) — local ablation showed that `unique_only` on its own captured the full signal.

---

## Local ablation result (what earns the "signs-of-life" claim)

Local proxy runs on a single GPU / MPS with short wallclock budgets (~15s / budget=0.8× / budget=1.2×), measuring `final_valid_bpb` on a held-out slice.

**Headline finding:** at `lookahead=16`, `unique_only` (and the equivalent `novelty_v1_lookahead_16`) beats `random_lookahead_16` on **6/6** of the (seed, budget) cells we ran. At shorter budgets the gain is small; at the slightly longer budget it is clearer. That budget-sensitivity is expected — a scheduler that reshapes the data distribution needs enough steps for the effect to show up in held-out loss.

Other things we learned (and why the submitted policy is what it is):

- `fifo_lookahead ≈ baseline` — having a lookahead buffer **on its own** is not the source of the gain; the *selection policy* matters.
- `random_lookahead_16 > baseline` — buffer + random pick already wins a bit, i.e. there's a real effect from *having* choice.
- `unique_only > random_lookahead_16` — on top of that, diversity-first selection gives an additional, consistent bump.
- `unique_only + transition` did not beat `unique_only` alone; `maxfreq_penalty` alone was worse than `unique_only`. So the composite `novelty_v1` collapses cleanly to its `unique` component.

Raw per-phase CSVs and the stability recheck are in `ablation_results/`.

Relevant columns across phases:

- `scheduler_phase1_results.csv` — first sweep across basic policies (`fifo_lookahead`, `random_lookahead`, `novelty_v1`) at a shared short budget.
- `scheduler_phase2_results.csv` — `lookahead ∈ {4, 8, 16}` sweep for `novelty_v1`.
- `scheduler_phase3_results.csv` — decomposing `novelty_v1` into `score_unique_only`, `score_transition_only`, `score_freq_penalty_only`, `score_unique_plus_transition`, `score_unique_minus_freq_penalty`.
- `scheduler_phase4_results.csv` — the (seed × budget) replication that produced the headline 6/6 claim: `baseline_no_scheduler` vs `novelty_v1_lookahead_16` at seeds `{42, 1337, 2024}` and budgets `{0.8, 1.2}`.
- `scheduler_stability_recheck_aggregate.csv` — aggregated stability replication across more seeds/budgets.

> ⚠️ **These are proxy numbers, not 8×H100 numbers.** `final_valid_bpb` here is ~3.3–3.6 because the proxy model and proxy validation slice are tiny and short. The ablation isolates the *relative ordering* between scheduler variants, not absolute BPB. We are submitting this as a non-record precisely because the headline BPB cannot be compared to the leaderboard.

---

## What we are asking compute for

A single clean first-experiment at real budget:

1. `baseline_full_train_8gpu` — the current `train_gpt.py` main path, all 8 ranks training.
2. `7_train_1_scheduler_unique_only_lookahead_16` — 7 training ranks, 1 scheduler rank, `SCHEDULER_POLICY=score_unique_only`, `SCHEDULER_LOOKAHEAD=16`.
3. (optional, if budget allows) `6_train_2_scheduler_unique_only_lookahead_16`.

We deliberately **are not** testing annealing schedulers, dynamic hardware reallocation, semantic preprocessors, LLM corpus rewriting, connector aux loss, or two-stage routers in this first-experiment. Those were either tried already in local ablations and judged not worth re-running at real budget, or belong to a later stage.

The experimental question this first-experiment answers is exactly one question:

> **Does `unique_only + lookahead=16` at a fixed 7:1 rank split beat the baseline on held-out BPB under the real 10-minute 8×H100 budget?**

If yes → the policy moves from "local signs of life" to "a real scheduler candidate" and we optimize the rank split / lookahead / scoring at real budget. If no → we learn that the proxy-to-real transfer was budget-sensitive in a way we had not captured, and we stop here.

---

## Files in this submission

- `README.md` — this file.
- `submission.json` — metadata (non-record; no 3-seed real-machine results yet).
- `ablation_results/` — all local proxy CSVs described above.
- `code/prototype_train_local_ablation.py` — the local ablation harness (`prototype/train.py` on our fork) that produced the `ablation_results/` CSVs. Supports `--data-scheduler-policy ∈ {none, fifo_lookahead, random_lookahead, novelty_v1, score_unique_only, ...}` and `--scheduler-lookahead-batches`.
- `code/train_gpt_scheduler_excerpt.py` — the `UniqueLookaheadScheduler` + `SchedulerBackedDistributedLoader` classes as wired into a modified `train_gpt.py` (the **real-machine path** intended for the grant run). Gated by `SCHEDULER_POLICY=score_unique_only`, `SCHEDULER_RANKS=1`, `SCHEDULER_LOOKAHEAD=16`.

## How to reproduce the local ablation

```bash
# From a checkout of this submission fork, plus prototype/train.py in the fork root:
python3 prototype/train.py \
    --data-scheduler-policy score_unique_only \
    --scheduler-lookahead-batches 16 \
    --max-wallclock-seconds 15 \
    --seed 42
# Repeat across seeds {42, 1337, 2024} and budgets {0.8, 1.2}, then compare to
# --data-scheduler-policy random_lookahead at the same settings.
```

## Lineage and credits

- Builds on the official `train_gpt.py` training loop (no model / tokenizer changes).
- Does **not** overlap with any existing leaderboard record — submitted as non-record.
- Companion submission: **Three-Layer Governance Probe + Cheap Gate v2 (Signs-of-Life)** (separate PR), which uses the same small-regime local-probe-then-real-machine pattern but on the inference-time governance side rather than the data-feeding side.

<\!-- ================================================================== -->
<\!-- GPT 填充区（线 A scheduler GPT）：你可以往下面补充 -->
<\!-- - 每一 (seed,budget) cell 的精确 BPB 差 (novelty - baseline) -->
<\!-- - phase1→4 的叙事衔接讲得更连贯 -->
<\!-- - 真机阶段的 accept/reject 判据（多少 seed、多少 step、多少 BPB 差算赢）-->
<\!-- - 不要加 records/... 之外的新 top-level 文件 -->
<\!-- ================================================================== -->
