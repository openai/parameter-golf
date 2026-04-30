# Opus — Parameter Golf Leaderboard Push

This folder is **Opus's working directory** for the OpenAI Parameter Golf challenge (16MB / 10-min track). It exists alongside Antigravity's work so the two agents don't trip over each other.

- **Antigravity's work**: `Antigravity/` (separate folder, not yet present)
- **Opus's work**: this folder

Opus and Antigravity work independently. Opus may peek at Antigravity's folder to look for ideas worth borrowing, but does not coordinate or hand off work.

## Current goal

Beat the standing SOTA of **1.0810 BPB** (PR #1493 by bigbag, 2026-04-09) by **≥0.005 nats** with 3-seed mean and p<0.01. Submission deadline: **2026-04-30**.

## Strategy in one sentence

Build directly on the PR #1493 SOTA stack. Push hard on **TTT improvements** (highest EV given the tight timeline) with **mixed-bit GPTQ** as a fallback angle. Keep `train_antigravity.py` alive in parallel as a non-record submission.

## Folder layout

```
Opus/
├── README.md          # this file — high-level status and pointers
├── PLAN.md            # 3-day execution plan with budget breakdown
├── DECISIONS.md       # log of why we chose / rejected directions
├── experiments/       # one .md per experiment (results, configs, logs)
├── notes/             # technical notes (SOTA architecture decode, etc.)
└── (any code we add)  # e.g. opus_train_gpt.py — variants we're testing
```

## Status

| Date | Phase | Status |
|------|-------|--------|
| 2026-04-27 | Pre-flight (CPU only) | ✅ SOTA decoded, selective-TTT patch written and locally validated, all 6 experiments pre-specified, pod setup + repro scripts staged. **Ready for GPU.** |
| 2026-04-27 | Day 1 reproduction | ⏳ Awaiting RunPod 1×H100 access |

## How to read this folder

- Start with `PLAN.md` for what we're doing and why.
- `experiments/` is the source of truth for what we've actually run. Each experiment file has: hypothesis, config, command, result, decision.
- `DECISIONS.md` is the audit trail for big calls (e.g. "killed mixed-bit GPTQ angle on Day 2 because TTT was tracking").
- Anything not in this folder is either Antigravity's or shared infrastructure.
