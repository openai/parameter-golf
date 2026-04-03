# Helix — Hypothesis

## Concept
Dual-stream co-firing: the crawler fires alongside every flat layer with
bidirectional cross-injection, instead of running sequentially after the encoder.

```
Flat:    F1 ──→ F2 ──→ F3 ──→ F4 ──→ ... ──→ F9
           ↘↗     ↘↗     ↘↗     ↘↗           ↘↗
Crawler: C  ──→ C  ──→ C  ──→ C  ──→ ... ──→ C
                                                ↓
                                          merge → output
```

Two intertwined strands — the flat stream builds unique representations while
the shared crawler continuously refines, and they cross-pollinate at every step.

## Parent
Ouroboros (BW XI): 1.13727008 BPB, 15,034,550 bytes

## ONE variable changed
`HELIX=1` — enables dual-stream co-firing mode.

## What changes architecturally
- Crawler fires 9 times (once per flat layer) instead of 2 (loop mode)
- Cross-injection via gated 32-dim projections (~65K new params):
  - flat→crawler: project flat hidden → 32d → expand into crawler residual
  - crawler→flat: project crawler hidden → 32d → expand into flat residual
- Final merge: `x = x_flat + sigmoid(gate) * x_crawl`
- All cross-injection up-projections zero-initialized (warm start = helix off)
- U-Net skip connections preserved on the flat stream
- Crawler uses shared weights at every step (same block, different input)

## What we expect
- The crawler sees progressively richer flat representations at every stage
- The flat stream gets continuous refinement feedback (not delayed to the end)
- More crawler passes (9 vs 2) = more refinement opportunities
- Trade-off: ~130-140ms/step (vs 100ms) = fewer total steps in 600s

## Gate arms
- `HELIX=1 HELIX_STRIDE=1` — crawler fires every flat layer (9 passes, ambitious)
- `HELIX=1 HELIX_STRIDE=3` — crawler fires every 3rd layer (3 passes, safe compute)

## Gate target
Signal above noise (−0.0003 BPB) at 1k steps. No blowups.
