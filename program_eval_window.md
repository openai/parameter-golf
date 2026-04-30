# Parameter Golf Research Program — Eval Window Lane

## Objective
Find record-track-viable sliding-window evaluation improvements that materially lower post-quantization `val_bpb` while staying inside a realistic eval-time budget.

## Primary Principle
The leaderboard now shows that overlapping-window evaluation is a major lever. Treat evaluation as a compression system, not just a final readout.

## What We Know
- Plain post-quant calibration already helped.
- Public leaders gained a large amount from sliding-window evaluation.
- Cheap overlap is more useful than fancy calibration if it improves context coverage enough.
- Recent competitive PRs suggest `stride=256` may slightly beat `64` while cutting eval cost substantially.

## Priority Order
1. Direct `stride=256` vs `64` vs `128` comparisons with score-only-new-token logic
2. Sliding-window evaluation with overlap and score-only-new-token logic
3. Stride tuning and batching for efficient overlap
4. Hybrid overlap + calibration combinations
5. Cache reuse or block reuse that preserves eval speed

## Preferred Directions
- Add overlapping eval windows with stride values like 256, 128, and 64
- Score only the fresh tokens rather than double-counting overlap
- Use center-token or right-edge scoring if it improves stability
- Reuse KV/cache state if it helps without changing semantics
- Combine windowing with existing temperature/softcap calibration only when clearly orthogonal

## Avoid
- Core-model architecture changes
- Tokenizer or dataset changes
- Eval methods that explode runtime or memory
- Pure calibration-only ideas unless they are directly supporting overlap
- Multi-phase or iterative calibration loops
- Adding many extra eval passes to an already expensive run
- Changes that require recompiling or restructuring the forward path

## Guidance
- Make one conceptual change at a time
- Prefer concrete window/stride logic over abstract eval heuristics
- Keep `eval_time_ms` visible and bounded
- Optimize for something that could plausibly transfer to an 8xH100 record submission
- The single most actionable question in this lane is whether `stride=256` beats `64` for our current stack
- Prefer tiny code diffs that change a constant, a scoring mask, or a windowing loop
- If a proposal would add more than a few eval passes, do not do it
- If the last few experiments timed out, become more conservative, not more ambitious
