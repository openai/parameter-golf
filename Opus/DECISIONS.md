# Decision Log

Audit trail for non-obvious calls. Each entry: date, decision, alternatives considered, reasoning. New entries at top.

---

## 2026-04-27 — Build on PR #1493 SOTA, not on `train_antigravity.py`

**Decision:** Use the SOTA file at `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py` as the base for the leaderboard push. Treat `train_antigravity.py` as a separate, parallel non-record submission.

**Alternatives considered:**
- (A) Push only the antigravity stack (MLA + 3.5× MLP + Int6 QAT, vocab 1024) for the record.
- (B) Bolt antigravity ideas (MLA, MTP) onto the SOTA stack from scratch.
- (C) Build directly on PR #1493 SOTA, keep antigravity as a side submission. **← chosen**

**Reasoning:**
- The antigravity stack is missing every layer of the current SOTA (no SP8192, no GPTQ-SDClip, no recurrence, no parallel residuals, no legal TTT). Reaching parity is weeks of work, not 3 days.
- The SOTA author chain is on the same code surface — adding to it is incremental and stylistically expected by the reviewers.
- Keeping antigravity alive as a non-record submission is ~free in compute and gives us a guaranteed submission either way.

---

## 2026-04-27 — Top angle: smarter TTT, not more architecture

**Decision:** Spend the bulk of compute exploring TTT variants (param-selective, chunk-size sweeps, momentum schedules) rather than architectural changes (MLA, MTP, depth changes).

**Alternatives considered:**
- New attention variant (MLA / linear / state-space hybrid) — requires retraining from scratch and weeks of tuning.
- More depth recurrence loops — already at 3 layers × 2 loops; diminishing returns.
- New optimizer (Lion, Sophia) — Muon is hard to beat at this scale.
- Mixed-bit GPTQ — kept as fallback (Day 2 pivot).

**Reasoning:**
- TTT is the newest layer in the SOTA stack (added in PR #549, refined through #1413, #1493). Less time for the community to optimize it.
- Current TTT is naïve: vanilla SGD on **all** params. A quantized model has only a small fp32 surface (`q_gain`, `attn_scale`, `mlp_scale`, `skip_weights`, `skip_gates`, `resid_mix`, `ln_scale_factor`); training only those is faster, lower-variance, and avoids fighting GPTQ rounding errors.
- TTT runs at eval time, so we can iterate on a fixed checkpoint without re-paying the 10-min training cost per experiment. This makes Day 2 triage 10× cheaper than retraining sweeps.
- Theoretical ceiling: TTT currently gives ~0.002 nats (1.0827 sliding → 1.0810 TTT). If we can extract another 0.005 from the same mechanism, that's our submission.
