# A/B Tests Round 3 — Post-1.1130 Optimization

## Current Best
**1.1130 val_bpb** | stride=76 | per-window SGD TTT | 14L | GPTQ int6 | EMA 0.997 | 575s eval | 15.87MB

## Best Clean PRs for Reference
| PR | BPB | Key technique |
|----|-----|--------------|
| #569 | 1.1175 | VRL + LeakyReLU² + Full GPTQ (no TTT) |
| #545 | 1.1179 | Int5 GPTQ + 33.6M model |
| #589 | 1.1178 | Soft-Round QAT + Backward-Looking TTT |
| #505 | 1.1181 | SwiGLU + VE128 (no TTT) |

---

## Experiments Queue

### 1. Value Residual Learning (VRL)
**Status: RUNNING (exp202_vrl)**
- Layer 0's V output blended into all subsequent layers via per-layer sigmoid gates
- Source: arxiv:2410.17897, PR #569 (1.1175 without TTT)
- Expected: -0.001 to -0.003 BPB
- Control: exp202_novrl (identical config, VRL_ENABLED=0)
- Extra params: 13 scalars (negligible)

### 2. QAT-Export Alignment
**Status: READY TO IMPLEMENT**
- Problem: STE fake-quant during training may use different clipping than GPTQ at export
- Fix: Match STE clip percentile to GPTQ's percentile (0.9995 from PR #569)
- Source: PR #569 explicitly aligns these
- Expected: -0.001 to -0.002 BPB (reduces quant gap)
- Risk: Low — one constant change

### 3. Soft-Round QAT
**Status: NEEDS IMPLEMENTATION**
- Replace hard `round()` in STE with temperature-controlled smooth approximation
- Source: PR #589 "Late Soft-Round QAT"
- Gives optimizer gradient signal near quantization bin boundaries
- Expected: -0.001 to -0.002 BPB
- Risk: Medium — new forward pass math, could destabilize training if temp schedule is wrong
- Implementation: `soft_round(x, temp) = x + (1/pi) * arctan(sin(2*pi*x) / temp)` or similar

### 4. Int5 GPTQ (fit bigger model)
**Status: NEEDS INVESTIGATION**
- Quantize to int5 (clip_range=15) instead of int6 (clip_range=31)
- Saves ~17% per weight → could fit 15th layer or wider MLP
- Source: PR #545 (33.6M params in 15.5MB with int5)
- Expected: depends on what we do with the space savings
- Risk: High — int5 quality loss might outweigh capacity gain at 14L
- A/B: int5 14L vs int6 14L (same architecture, just quant precision)

### 5. Backward-Looking Chunk TTT
**Status: IMPLEMENTED (exp201 chunked), already tested**
- Score chunk N, train on chunks 0..N-1 (not on N itself)
- Result: 1.1175 BPB in 138s (faster but 0.005 worse than per-window)
- Verdict: Per-window SGD is better for us. Could revisit with AdamW + more epochs.

### 6. EVAL_STRIDE=32
**Status: NOT VIABLE (time budget)**
- PR #545 uses stride=32 for finer overlap
- Our stride=76 at 575s leaves no room for stride=32 (~1300s est)
- Only viable if combined with chunked TTT (~138s base) + stride=32 scoring
- A/B: chunked TTT stride=32 vs per-window SGD stride=76

### 7. MHA 8/8 (drop GQA)
**Status: NEEDS TESTING**
- PR #545 uses full multi-head attention (8/8) vs our GQA (8/4)
- More attention capacity but more params per layer
- At 14L this might push artifact over 16MB unless combined with int5
- A/B: 14L 8/8 MHA vs 14L 8/4 GQA

### 8. Early QAT (threshold 0.5)
**Status: NEEDS TESTING**
- PR #545 starts QAT at 50% through warmdown (threshold=0.5)
- Our current: no QAT during training
- More QAT steps = model better adapted to quantization noise
- Source: PR #414 uses 0.15, PR #545 uses 0.5
- A/B: QAT threshold 0.5 vs 0.15 vs 0 (current)

### 9. Magnitude Pruning Post-GPTQ
**Status: NEEDS TESTING**
- PR #569 uses 2% magnitude pruning AFTER quantization for compression
- We disabled pruning (PRUNE_FRAC=0)
- Could help compression → smaller artifact → room for larger model
- A/B: PRUNE_FRAC=0.02 vs 0 (current)

### 10. Temperature Search (post-TTT)
**Status: TESTED — T=0.98 HURT (+0.0002)**
- Verdict: Not useful for our SGD TTT setup

---

## Completed Results

| Experiment | BPB | Time | Delta vs baseline | Notes |
|-----------|------|------|-------------------|-------|
| stride=64 per-window SGD | 1.1126 | 654s | — | Over time budget |
| stride=68 | 1.1129 | 640s | +0.0003 | Over budget |
| stride=72 | 1.1129 | 604s | +0.0003 | 4s over |
| **stride=76** | **1.1130** | **575s** | **+0.0004** | **Submission candidate** |
| stride=80 | 1.1130 | 547s | +0.0004 | Safe margin |
| stride=96 | 1.1131 | 458s | +0.0005 | Very safe |
| stride=128 | 1.1133 | 347s | +0.0007 | Very safe |
| stride=76 T=0.98 | 1.1132 | 566s | +0.0006 | Temp hurts |
| chunked AdamW 1ep | 1.1175 | 138s | +0.0049 | Fast but worse |
| VRL (exp202) | pending | ~575s | ? | Running |
| no-VRL control | queued | ~575s | ? | After VRL |

---

## NEW from research (March 24)

### 11. Multi-Pass Score-First TTT
**Status: NEEDS INVESTIGATION**
- PR #573 uses 3 independent adaptation trajectories with shifted data orderings
- For each token, take min(NLL) across passes — best-of-3 scoring
- Claims 1.0523 BPB (need to verify legality)
- Expected: significant gain if legal
- Risk: 3x eval time — would need stride increase to compensate

### 12. Full GPTQ (not lite)
**Status: NEEDS TESTING**
- PR #593 claims full GPTQ improves post-quant BPB by 0.0048 vs GPTQ-lite
- We already use GPTQ but need to verify it's the full Hessian version with Cholesky error compensation
- Check: does our GPTQ_ENABLED=1 do full Cholesky or simplified?

### 13. EMA Decay Tuning
**Status: NEEDS TESTING**
- Our 14L model is deeper than typical 11L — may need different EMA dynamics
- Test: EMA(0.995), EMA(0.9975), EMA(0.9985) vs current EMA(0.997)
- Quick eval-only A/B if we save pre-EMA checkpoints

### 14. Weight Sharing (Recursive Depth)
**Status: HIGH RISK, INVESTIGATE LATER**
- PR #579: 6 unique blocks × 2 loops = 12 effective depth, wider MLP
- GPTQ catastrophically fails at 3+ loops — only 2 loops viable
- Would require major architecture rework
- Potential: large gain but untested at our scale

### 15. Cosine-Annealed TTT LR
**Status: EASY TO TEST**
- PR #581, #589: cosine LR during TTT (not flat) improves by 0.003+
- We use flat lr=0.002 — easy to add cosine decay across batches
- Can test with saved model (eval-only mode)

---

## Priority Order (Updated)
1. **VRL** (running) — architecture change, proven in #569
2. **QAT-export alignment** — low effort, directly attacks quant gap
3. **Cosine-annealed TTT LR** — easy eval-only test, proven in #581/#589
4. **Early QAT threshold 0.5** — proven in multiple PRs
5. **Soft-round QAT** — novel, higher effort but targets quant gap
6. **Full GPTQ verification** — make sure we're doing full Hessian, not lite
7. **Int5 + wider MLP** — risky but could unlock more capacity
8. **Multi-pass TTT** — need to verify legality and time budget
9. **EMA decay tuning** — quick eval-only if checkpoints available
10. **Pruning 2% post-GPTQ** — easy test, helps compression
