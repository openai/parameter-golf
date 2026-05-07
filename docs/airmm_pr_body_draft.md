## Summary

This PR adds an experimental non-record submission exploring **Adaptive Importance Routing Memory Model (AIR-MM)** — a lightweight mechanism for selective compute allocation in parameter-constrained transformers.

Instead of treating every token equally, AIR-MM estimates token importance and uses that signal to gate residual updates, so the model can focus limited capacity on higher-value information. The goal is not just to shrink a standard transformer, but to introduce a genuinely different mechanism for deciding where updates matter most.

## What was added

- `records/track_non_record_16mb/2026-04-08_AIRMM_v2_Debug_Prototype/` — full submission folder containing:
  - `train_gpt.py` with AIR-MM v2 implementation (learned importance gate + positional recency signal + lightweight combiner)
  - `README.md` documenting the architecture, motivation, and local results
  - `submission.json` with conservative metadata
  - `requirements.txt`
  - `train_debug_v2.log` from a local CPU fast-debug run
- `docs/airmm_architecture.md` — internal architecture document

## What is novel

AIR-MM adds a per-token, per-layer importance gate to the residual stream. A small two-layer MLP (~800 params/block) scores each token's importance from its hidden representation. In v2, a positional recency signal is blended in via a tiny learned combiner (3 params/block). The combined gate scales both attention and MLP residual updates multiplicatively, with a min-scale floor to prevent token silencing.

This is distinct from pruning (no tokens dropped), MoE (no discrete routing), dropout (not random), and early exit (no layers skipped). Every token is still processed; the model learns how much weight each update carries.

## Evidence so far

Local FAST_DEBUG results (CPU, 128-dim, 4-layer, 10 steps):

| Config | val_bpb (step 10) | train_loss (step 10) | Params |
|---|---|---|---|
| AIR-MM v1 (learned gate) | 4.0261 | 6.8806 | 608,656 |
| AIR-MM v2 (learned + recency) | 4.0254 | 6.8788 | 608,668 |

- Training is stable, no divergence
- Baseline is fully preserved when `USE_IMPORTANCE_ROUTING=0`
- Gate learns non-trivial distributions (learned ~0.46, combined ~0.62, recency spans [0, 1])
- Small directional improvement from v1 to v2

**These are CPU fast-debug numbers only.** They are not comparable to leaderboard scores and do not constitute evidence of competitive performance. GPU-scale validation is needed.

## Why non-record / experimental track

This submission is positioned as an experimental direction, not a performance claim. The mechanism is active and stable, but:
- Only tested on a reduced debug model (128-dim, 4 layers vs full 512-dim, 9 layers)
- Only 10 training steps (vs thousands in a real run)
- CPU-only, no GPU validation yet
- The directional improvements are small and could be noise at this scale

The purpose of this PR is to make the approach visible to the community, get feedback on the mechanism design, and establish a baseline for future GPU-scale experiments.

## Limitations

- Gate behavior is relatively uniform across blocks at debug scale — unclear if this is a scale artifact or a fundamental limitation
- Recency signal is a simple positional ramp; more sophisticated signals (surprise, context-shift) are designed but not yet implemented
- No quantization or compression testing — unknown how the gate interacts with int8+zlib roundtrip
- Parameter overhead analysis only done at debug scale

## Next steps

- GPU-scale run with full 9x512 configuration
- Controlled ablation: routing ON vs OFF under identical conditions
- Hyperparameter sweep: min_scale, recency_strength, hidden_dim
- Quantization compatibility testing
- v3: additional importance signals (surprise, context-shift)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
