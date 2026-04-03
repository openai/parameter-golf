# Helix Micro Test Results — DGX Spark

Date: 2026-04-03
Config: dim=256, seq=512, 200 steps, compile=off, seed=444

## Full Results Table

| Arm | Config | BPB | vs 5F ctrl | step_ms | params |
|-----|--------|-----|------------|---------|--------|
| **Phase 1: Foundation** |
| A0 | 3F ctrl (no helix) | 1.9044 | — | 515 | 2,949,652 |
| A1 | 3F helix s1 + 2loop | 1.9223 | — | 631 | 2,966,292 |
| A2 | 3F helix s1 + 1loop | 1.9144 | — | 629 | 2,954,004 |
| **Phase 2: Depth** |
| B0 | 5F ctrl (no helix) | 1.9080 | baseline | 690 | 4,131,612 |
| B1 | 5F helix s1 + 2loop | 1.9102 | +0.0022 | 1014 | 4,148,252 |
| B2 | 7F helix s1 + 2loop | 1.8972 | −0.0108 | 1407 | 5,330,212 |
| **Phase 3: Stride** |
| C0 | 5F helix s1 1loop (5 fires) | 1.9065 | −0.0015 | 1003 | 4,135,964 |
| C1 | 5F helix s2 1loop (2 fires) | 1.9056 | −0.0024 | 681 | 4,135,964 |
| C2 | 5F helix s5 1loop (1 fire) | **1.8973** | **−0.0107** | **586** | 4,135,964 |
| **Phase 4: Bridge Width** |
| D0 | 5F helix dim=8 | 1.9103 | +0.0023 | 986 | 4,127,772 |
| D1 | 5F helix dim=16 | 1.9067 | −0.0013 | 980 | 4,135,964 |
| D2 | 5F helix dim=32 | 1.8987 | −0.0093 | 980 | 4,152,348 |
| D3 | 5F helix dim=64 | **1.8616** | **−0.0464** | 984 | 4,185,116 |
| **Phase 5: Interactions** |
| E0 | 5F helix + inst32 | 1.9061 | −0.0019 | 982 | 4,160,540 |
| E1 | 5F helix + anchor16 | 1.9064 | −0.0016 | 985 | 4,164,636 |
| E2 | 5F helix + tap16 | 1.9106 | +0.0026 | 982 | 4,152,348 |

## Key Findings

### 1. BRIDGE WIDTH IS THE DOMINANT LEVER (Phase 4)
dim=64 achieved **1.8616 BPB** — a massive −0.0464 vs control. The improvement is
monotonic: 8 → 16 → 32 → 64 each step is a clear gain. And dim=64 costs almost
nothing extra in compute (984ms vs 980ms for dim=16) or params (+53K).

### 2. LESS FREQUENT FIRING IS BETTER (Phase 3)
stride=5 (1 fire) beats stride=1 (5 fires): 1.8973 vs 1.9065. AND it's nearly 2× faster.
Rare but powerful cross-injection > frequent weak injection. The crawler doesn't need
to fire every layer — one well-timed exchange carries more signal.

### 3. HELIX + SEQUENTIAL LOOPS = WASTEFUL (Phase 1-2)
A1 (helix + 2loop) is the worst helix arm. B1 (5F helix + 2loop) is worse than
C0 (5F helix + 1loop). The helix IS the recurrence — stacking additional sequential
loops on top creates redundancy and gradient conflict.

### 4. EXISTING FEATURES DON'T STACK (Phase 5)
inst32, anchor16, tap16 — all neutral or slightly negative with helix. The helix
cross-injection subsumes what these mechanisms were trying to do.

### 5. HELIX NEEDS DEPTH (Phase 1 vs 2)
At 3F, helix hurts. At 5F, it's neutral-to-positive. At 7F, strong signal. The
cross-injection needs sufficient flat layer diversity to be worth routing.

## Implied Optimal Config
- **HELIX=1, HELIX_DIM=64 (or higher?), HELIX_STRIDE=5 (rare firing), CRAWLER_LOOPS=1**
- Deep flat stack (7F+), single crawler loop, one powerful cross-injection per stride group
- No inst/anchor/tap needed — helix handles the cross-stream communication

## Next Steps
1. Test dim=128 (does the monotonic trend continue?)
2. Test Marco-Polo cross-attention at dim=64 (content-addressed routing)
3. Test at full scale (dim=512, 9F, 8×H100)
4. Missing: 7F control (no helix) to isolate helix signal from depth signal
