# Parameter Golf Issue #140 - Overview

**Issue:** [Parameter Golf Live AI Commentary + Analysis / Ideas](https://github.com/openai/parameter-golf/issues/140)
**Author:** @notapplica | **Last Updated:** Mar 21, 3:02 AM PT

## Competition Summary

- **Goal:** Train the best language model in a **16MB artifact**, under **10 min on 8xH100s**
- **Metric:** Bits per byte (BPB) on FineWeb validation set — lower is better
- **Baseline:** 1.2244 BPB
- **Current Official SOTA:** 1.1428 BPB (@thwu1, PR #180)
- **Best Pending:** 1.1250 BPB (@jfprincz, PR #315)
- **Progress:** ~0.10 BPB improvement in ~2.5 days, 332+ PRs submitted

## Key Rules

- Artifact <= 16,000,000 bytes (code + compressed model)
- Training <= 10 min on 8xH100 SXM; Eval <= 10 min (separate)
- No network calls during eval
- New SOTA must beat current best by >= 0.005 nats at p < 0.01 (typically 3 seeds)
- Tokenizer-agnostic (BPB normalizes across tokenizers)
- Val tokens cannot be stored in artifact (paid prefix ruled out)
- Only backward-looking TTT allowed (adapt on already-scored tokens)

## File Index

| File | Contents |
|------|----------|
| [01-leaderboard.md](01-leaderboard.md) | Official + pending leaderboard tables |
| [02-core-techniques.md](02-core-techniques.md) | The 5 foundational techniques used by all competitive entries |
| [03-advanced-techniques.md](03-advanced-techniques.md) | SmearGate, XSA, TTT, #315's innovations, and more |
| [04-what-doesnt-work.md](04-what-doesnt-work.md) | Negative results and failure patterns |
| [05-untried-ideas.md](05-untried-ideas.md) | Ranked untried combinations by expected value |
| [06-tier-analysis.md](06-tier-analysis.md) | What separates each performance tier |
| [07-idea-lineage.md](07-idea-lineage.md) | Technique origins and adoption tracking |
