# Stage 2 H100 Matrix R1

This is the concrete `8xH100` Stage 2 experiment matrix.

Purpose:

- make Stage 2 executable as a set of exact configs,
- separate ready-to-run slots from slots that still require code patching,
- build on both:
  - the public record frontier,
  - and our derivative internal winners.

## Ground Rules

- `S2-B0` is mandatory.
- `S2-E1` to `S2-E4` are the first wave.
- `S2-E5` and `S2-E7` are intentionally marked `needs_patch`.
- every training-side run should be read with:
  - fixed-chunk result
  - sliding result when available
  - `step_avg`
  - steps reached
  - pre/post-quant gap

## Slot Map

| Slot | Role | Status | Core question |
| --- | --- | --- | --- |
| `S2-B0` | trunk gate | ready | can we reproduce the clean public training trunk? |
| `S2-E1` | eval extension | ready | does sliding eval still help on the seq4096 trunk? |
| `S2-E2` | export extension | ready | does fp16 tied embedding help on the seq4096 trunk? |
| `S2-E3` | export extension | ready | does our quant-quality win transfer to real `8xH100`? |
| `S2-E4` | public-recipe transfer | ready | does Muon WD help on top of the strong trunk? |
| `S2-E5` | public-recipe transfer | needs_patch | does overtone / phase-structured init transfer? |
| `S2-E6` | throughput derivative | ready | does adaptive Muon reduce `ms/step` enough to win on the trunk? |
| `S2-E6B` | record-derivative branch | ready | can record-like `2048` plus our throughput winner improve the public path? |
| `S2-E7` | architecture challenge | needs_patch | does 10-layer only work with the full support stack? |
| `S2-E8` | schedule challenge | ready | does the warmdown-heavy branch work when tested deliberately? |

## Why This Matrix

The matrix deliberately splits into three kinds of slots:

- attribution on the clean public trunk
- derivative “record + our winners” hybrids
- deferred branch tests that are only worth running if the first waves justify them
