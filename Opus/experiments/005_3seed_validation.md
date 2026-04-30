# Experiment 005 — 3-seed validation on 8×H100

**Date:** TBD (Day 3)
**Hypothesis:** The Day 2 winning config beats SOTA `val_bpb = 1.0810` by ≥0.005 nats with 3-seed mean and p<0.01.
**Baseline:** PR #1493: 3-seed mean 1.0810, std 0.0002
**Cost:** 3 × full 8×H100 runs ≈ $40 of compute (~12 min each)

## Decision rule before this experiment

We only run this if:
- Experiment 002 produced a filter that **single-seed beat the baseline by ≥0.002 nats**, AND
- Experiments 003+004 stacked at least **0.003 nats** above the baseline single-seed.

Otherwise, the 3-seed mean is unlikely to clear 0.005, and we pivot remaining budget to mixed-bit GPTQ (Experiment 010) or polish the antigravity non-record submission.

## Setup

The full leaderboard pipeline: train from scratch + GPTQ + sliding + TTT, all in one run, on 8×H100 SXM. Same image/data as the published SOTA.

Each seed gets its own 8×H100 pod ($20/hr × ~13 min = ~$4.50 + ~$5 pod overhead).

## Commands

```bash
# Per-seed run on 8×H100 SXM
for SEED in 42 314 999; do
  TTT_ENABLED=1 TTT_PARAM_FILTER=$WINNER_FILTER \
    TTT_LR=$WINNER_LR TTT_EPOCHS=$WINNER_EP TTT_CHUNK_TOKENS=$WINNER_CHUNK \
    SEED=$SEED \
    QK_GAIN_INIT=5.25 \
    RUN_ID=opus_e005_seed${SEED} \
    torchrun --standalone --nproc_per_node=8 \
      Opus/code/train_gpt_v1.py 2>&1 | tee Opus/experiments/logs/005_seed${SEED}.log
done
```

## Result

| Seed | `val_bpb_sliding` | `val_bpb_ttt` | Artifact bytes |
|------|-------------------|---------------|----------------|
| 42   | | | |
| 314  | | | |
| 999  | | | |
| **Mean** | | | |
| **Std**  | | | |

## Acceptance criteria

For an accepted record submission to PR `openai/parameter-golf`:

1. **3-seed mean of `val_bpb_ttt` ≤ 1.0760** (SOTA - 0.005)
2. **All 3 individual seeds ≤ 1.0790** (within ~0.003 nats of each other)
3. **Std ≤ 0.0007** (so p < 0.01 by one-sided Welch's t-test vs. SOTA's 1.0810)
4. **All 3 artifacts ≤ 16,000,000 bytes**
5. **All 3 train wallclocks ≤ 600s**
6. **All 3 eval wallclocks ≤ 600s** (sliding + TTT combined)

## Failure modes & contingency

- **Mean clears 1.0760 but std is high (>0.0007):** run 3 more seeds {7, 11, 13} on remaining budget. If 6-seed mean still clears, submit.
- **Mean lands 1.0760–1.0780:** doesn't clear 0.005-nat threshold. Submit as a non-record, well-documented PR — still on the public record.
- **Mean lands above 1.0780:** the Day 2 win didn't replicate. Acknowledge in the antigravity non-record submission instead.

## Submission

If accepted, the next-day work is:
- Write `records/track_10min_16mb/2026-04-29_SelectiveTTT/README.md` modeled on the PR #1493 README
- Generate `submission.json` with the 3 seed numbers
- Bundle the modified `train_gpt.py` (re-golfed and LZMA-wrapped to fit under 16MB *with* the tuned model)
- Open the PR on `openai/parameter-golf` from `GodlyDonuts/parameter-golf:claude/busy-thompson-9c94f9`
