# Parameter Golf Research Program

## Objective
Minimize validation bits-per-byte (BPB) on FineWeb, subject to:
- 16MB total artifact size (code + compressed int8+zlib weights)
- 10 minutes training on 8×H100s
- Scored on post-quantization roundtrip BPB

Current frontier: **1.3508 BPB** (core_promotion experiment 12, 600s 1×H100, validated)
- Config: 12 layers, 448-dim, mlp_mult=2, matrix_lr=0.08, warmdown_iters=600, ~17.3M params
- Artifact: 15.8MB (headroom: ~0.2MB — very tight)
- Quantization gap: 0.0007 (essentially zero — this architecture quantizes perfectly)
- Steps reached: ~1061 in 600s

Previous 12×512 result (over-budget reference): 1.3346 BPB at 20MB — proves extra width helps but doesn't fit.

8×H100 10-min baseline: 1.2244 BPB (9-layer, 512-dim, 4 KV heads, tied embeddings, ~15.8MB)

## Key Insight
This is a compression challenge. The question is: how much language understanding can you pack
into 16MB of int8+zlib compressed weights, trained in 10 minutes?

Three levers: (1) better architecture per parameter, (2) better training per step,
(3) better compression per parameter.

## What We've Learned (1×H100 Proxy)

### Strong signals
- **Depth helps enormously**: adding layers has consistently improved BPB. At 600s, a 12-layer model hit 1.3346 BPB (vs 1.4981 at 180s) — depth is by far the strongest lever.
- **Shorter warmdown helps a lot**: 1200→800→600 each gave clear wins. The default 1200 is way too long for short runs.
- **matrix_lr increase helped**: 0.04→0.06 was a clear win; current best runs at ~0.07. But 0.08→0.09 did NOT help (two attempts, both worse).
- **Quantization gaps are tiny** (~0.002 or less on the best configs, and 0.0007 on the 600s run) — no need to fear int8.

### Weak or negative signals
- **scalar_lr increase did NOT help** (1.5574→1.5801, reverted)
- **matrix_lr=0.09 did NOT help** (two attempts: 1.5148 and 1.5029, both worse than 1.4967)
- **Naive mlp_mult increases did NOT help** when model already has mlp_mult=3
- **Generic activation swaps** (SwiGLU/GELU) were worse than ReLU² on prior proxy runs
- **warmdown_iters=400 caused artifact over-budget** (16.2MB) — the schedule change affected weight distribution

### Current regime
- The **12×448 dim, mlp_mult=2** config is the validated frontier at 1.3508 BPB, 15.8MB
- Artifact headroom is very tight (~0.2MB) — any param increase risks going over budget
- Quantization gap is essentially zero (0.0007) — the architecture quantizes perfectly
- The path forward is **local refinement** around this shape, not large architectural changes

## Research Priorities (ordered by expected value)

**The validated frontier is 12×448, mlp_mult=2 at 1.3508 BPB. Refine locally around this base.**

### Priority 1: Schedule refinement (highest expected value, zero size risk)
1. **Warmdown sweep around 600** — try 500, 550, 700. The warmdown→BPB trend has been strong; find the local optimum for this shape at 600s.
2. **Narrow matrix_lr sweep around 0.08** — try 0.07, 0.075, 0.085. The optimal LR may have shifted with the new shape.

### Priority 2: Shape refinement (medium expected value, watch artifact size)
3. **Width nudges around 448** — try 432, 464, 480. Artifact is at 15.8MB, so increases must be small. 464 might fit; 480 probably won't.
4. **KV-head reduction** — try num_kv_heads=2 (instead of 4) to save params for a slight width or mlp increase.
5. **mlp_mult exploration** — try mlp_mult=3 at dim=448 (may be too large) or fractional-MLP tricks if available.

### Priority 3: Depth/width rebalancing (only if it stays under cap)
6. **13 layers with aggressive width reduction** — e.g. 13×416 or 13×400 if it fits under 16MB.
7. **Light parameter tying** — share weights across adjacent layers for more effective depth for free.

### DO NOT TRY (explicitly de-prioritized)
- scalar_lr experiments (already shown to hurt)
- matrix_lr >= 0.09 (two attempts, both worse at the old shape)
- generic activation function swaps (SwiGLU, GELU — already tested, worse)
- large multi-change jumps (we are in a local-refinement regime)
- naive depth increases without a concrete size-recovery plan (12×512 was 20MB over budget)

## Experiment Guidelines
- Make ONE change per experiment so we can isolate effects
- Always explain the hypothesis: why should this help?
- On the 1xH100 proxy, prefer changes that preserve or improve steps/sec
- Monitor artifact size — current best is at ~13.6MB with ~2.4MB headroom
- If over 16MB, the experiment only matters if it reveals a path to recover compression
- Track both pre-quantization and post-quantization BPB; some changes learn better but quantize worse
- Failed experiments are valuable — note what didn't work and why
- When a hyperparameter sweep finds an optimum, combine it with the best known config
- **Follow the priority order above** — try higher-priority changes before lower-priority ones
- Do NOT repeat experiments that have already failed in the history
- Prefer small, interpretable changes that build on the current best incrementally
