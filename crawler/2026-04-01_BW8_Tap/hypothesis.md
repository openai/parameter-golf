# BW8_Tap — Hypothesis

**Parent:** BW5 (1.18672385 BPB, 8.61MB)
**One variable:** `CRAWLER_TAP_DIM=32 CRAWLER_TAP_LOOP_SPECIFIC=0` (shared encoder tap)

## Mechanism

Project intermediate encoder layer outputs (shallow + deep) once into a 32-dim
tap embedding. All 3 crawler loops inject the same shared projection of these
frozen encoder signals into their residual. The encoder tap is computed once
before looping — negligible overhead.

The tap provides a stable, pre-quantization anchor — the signal the crawler
is supposed to be refining. All loops read the same anchor (shared vs per-loop).
Zero-init on `shared_tap_up` → warm start near BW5 behavior.

## Gate evidence (BW7 MegaGate, 4×GPU SDPA, 2000 steps)

| Arm | int6_sw_bpb | delta |
|-----|-------------|-------|
| CTRL-00 (baseline) | 1.28912666 | — |
| TAP-02 (per-loop dim=32) | 1.28772880 | −0.00140 |
| **TAP-03 (shared dim=32)** | **1.28560427** | **−0.00352** |
| TAP-04 (per-loop dim=16) | 1.28646248 | −0.00266 |

TAP-03 is the strongest signal in the MegaGate. −0.00352 is 12× the variance
noise floor. Shared outperforms per-loop at this dim — one general anchor beats
3 specializing at dim=32. SDPA + 4×GPU environment, so absolute values don't
transfer — relative delta is the signal.

## Gate target (proper 8×GPU FA3)

Beat control arm at 2000 steps with step_avg ~74ms.
Promote to full run if delta holds above noise floor (~0.0003 BPB).

## Full run target

Beat **1.18672385 BPB** (BW5 champion, seed=444) with artifact ≤ 16MB.
