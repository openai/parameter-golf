# bandit_wagon_tap — Per-Loop Gated Encoder Tap

## Background

The crawler loops 3× over the same bottleneck, starting only from the final encoder
output. Shallower encoder layers — which captured low-level patterns, raw token features,
and early abstractions — never reach the crawler directly. That signal "passes by" and
is only available to the decoder via U-Net skip connections.

Under quantization, the crawler accumulates depth error across loops. It has no stable
reference to cross-check against: the FLOW mechanism is self-referential (current x →
correction), and XSA is stateless (recomputed fresh each loop). Nothing anchors the
crawler to the original, unquantized encoder signal.

**Hypothesis:** Giving the crawler a frozen tap into intermediate encoder representations
— projected to a small tap_dim — will reduce quantization drift by providing a stable
reference signal that each loop can consult. Per-loop specificity allows loop 0 (early
abstraction) to listen differently from loop 2 (deep refinement).

## Architecture

```
Encoder layer 0 out ──── tap_proj[0] (512→tap_dim) ─┐
Encoder layer 1 out ──── tap_proj[1] (512→tap_dim) ─┤──cat──► [B,T,tap_dim*2]
                                                     ┘         │
                                               loop_tap_up[loop] (per-loop, zero-init)
                                                               │
                                               x_loop += tap_inject[loop]
```

**tap_proj**: shared across loops — encode the "essence" of each encoder layer once
**loop_tap_up**: per-loop (or shared) — each loop learns which essence it needs

Tap signal is computed once before the loop starts from frozen encoder outputs.
Zero-init on loop_tap_up → warm start identical to current behavior.

## Why this is different from FLOW

| | FLOW | Encoder Tap |
|--|------|-------------|
| Signal source | Current x (self-referential, drifts with quant error) | Frozen encoder outputs (never re-quantized) |
| Loop specificity | Yes (loop_inst_up[loop]) | Yes (loop_tap_up[loop]) |
| Anchoring | None — FLOW tracks drift | Yes — tap is the pre-drift reference |
| Overhead | 2 matmuls/loop (proj + up) | proj once + 1 matmul/loop |

## Implementation Note

The existing skip connections in `_run_encoder` already capture intermediate encoder
outputs in `skips`. These are passed as `enc_outputs` to `_run_crawler` with no extra
compute. `tap_proj` projections run once, then each loop uses `loop_tap_up[loop]`.

## Arms

| ID | TAP_DIM | LOOP_SPECIFIC | TAP_LAYERS | Params added | Purpose |
|----|:-------:|:-------------:|:----------:|:------------:|---------|
| BWT-00 | 0 | — | — | 0 | **Control repin** — must match BW2-00 ±0.002 |
| BWT-01 | 32 | shared | all | ~99K | Does any tap help? Simplest version |
| BWT-02 | 32 | per-loop | all | ~131K | **Core hypothesis** — loop-differentiated listening |
| BWT-03 | 16 | per-loop | all | ~66K | Less essence — find minimum useful bottleneck |
| BWT-04 | 64 | per-loop | all | ~263K | More essence — does richness matter? |
| BWT-05 | 32 | per-loop | deep only | ~82K | Deep encoder only — is shallow useful? |
| BWT-06 | 32 | per-loop | shallow only | ~82K | Shallow encoder only — is raw signal the key? |

*Params = tap_proj + loop_tap_up. For dim=32, all layers: 2×512×32 + 3×64×512 = 32K + 99K = 131K*

## Decision Rules

**Gate 0 — control repin (BWT-00):**
Must land 1.521–1.526. If it misses: code bug. Stop.

**Gate 1 — signal present:**
Any arm must beat BWT-00 by ≥0.005. If none do: tap is not a useful lever at proxy scale.

**Gate 2 — promotion:**
Winning arm → 2000-step gate → combine with XSA=15 + winning choke + winning smear
before 8×H100 full run.

**Key comparisons:**
- BWT-01 vs BWT-02: does per-loop differentiation add value over shared?
- BWT-02 vs BWT-05: does shallow encoder add anything beyond deep alone?
- BWT-02 vs BWT-06: is the raw (shallow) signal the actually useful part?
- BWT-03 vs BWT-04: where is the tap_dim sweet spot?

## Locked Base Config

| Setting | Value | Source |
|---------|-------|--------|
| `NUM_FLAT_LAYERS` | 4 | BW5F confirmed |
| `XSA_LAST_N` | 11 | baseline |
| `MODEL_DIM` | 512 | BW anchor |
| `CRAWLER_LOOPS` | 3 | CL1 |
| `CRAWLER_MLP_MULT` | 6.0 | CL3 |
| `CRAWLER_MLP_CHOKE_DIM` | 0 | isolate tap variable |
| `CRAWLER_LOOP_SMEAR` | 0 | isolate tap variable |
| `CRAWLER_MLP_LEAKY_SLOPE` | 0.5 | control value |
| `SEED` | 444 | BW ablation |

## Results

| ID | DIM | LOOP | LAYERS | Step avg (ms) | Raw val_bpb | INT6_SW_BPB | Quant gap | Delta |
|----|:---:|:----:|:------:|:-------------:|:-----------:|:-----------:|:---------:|:-----:|
| BWT-00 | 0 | — | — | TBD | TBD | TBD | TBD | control |
| BWT-01 | 32 | shared | all | TBD | TBD | TBD | TBD | TBD |
| BWT-02 | 32 | per-loop | all | TBD | TBD | TBD | TBD | TBD |
| BWT-03 | 16 | per-loop | all | TBD | TBD | TBD | TBD | TBD |
| BWT-04 | 64 | per-loop | all | TBD | TBD | TBD | TBD | TBD |
| BWT-05 | 32 | per-loop | deep | TBD | TBD | TBD | TBD | TBD |
| BWT-06 | 32 | per-loop | shallow | TBD | TBD | TBD | TBD | TBD |

Reference: BW2-00 (no tap, XSA=11) → 1.52365
