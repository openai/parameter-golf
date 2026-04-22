# Record: SP8192 + CaseOps + Gated Attention + Quant Gate + Loop4-5 + Phased TTT + Frozen Recurrent Alpha — val_bpb 1.06421

**val_bpb = 1.06421** (3-seed mean, std 0.00023) | **~15.98 MB** | 8×H100 SXM, 600s train / 600s eval | Phased TTT

- Stacks 2 new techniques on the SP8192 CaseOps base (#1736): **frozen learned recurrent alpha/beta** (cross-layer blend scalars trained to convergence then frozen, `RECUR_ALPHA_ENABLED=1 NUM_LOOPS=2`), and **LoRA-TTT improvements** (warm-start A, alpha=144, WD=1.0) from PR #1767
- **−0.00128 BPB** vs base submission PR #1736 (1.06549)
- CaseOps legality pending issue #1604

## 3-Seed Results

| Seed | Pre-TTT BPB | **TTT BPB** | val_loss (nats/tok) | Artifact | Train | Eval |
|------|------------:|------------:|---------------------|----------|------:|-----:|
| 1    | 1.07704     | **1.06395** | 2.32833             | 15,976,882 | 596.1s | 458.7s |
| 777  | 1.07737     | **1.06429** | 2.32906             | 15,975,842 | 596.1s | 458.9s |
| 2025 | 1.07742     | **1.06438** | 2.32927             | 15,976,882 | 596.1s | 453.4s |
| **Mean** | **1.07728** | **1.06421** | **2.32889** | **15,976,535** | 596.1s | 457.0s |
| **Std**  | 0.00021     | **0.00023** | | | | |

All 3 seeds clear both 600s budgets (train + eval) and the 16,000,000-byte decimal artifact cap.

## Key Techniques

1. **SP8192 CaseOps** — Lossless reversible case normalization (TITLE/ALLCAPS/CAPNEXT/ESC operators). Pending #1604.
2. **GatedAttn + QuantGate** (PR #1736) — Full-dim attention gate with int8 passthrough.
3. **Loop4-5 depth recurrence** (PR #1736) — `NUM_LOOPS=2`, recurrence on layers 3–5.
4. **Frozen Recurrent Alpha/Beta** *(this PR)* — Learnable cross-layer blend scalars (`RECUR_ALPHA_ENABLED=1`) trained to convergence then frozen as constants. Converged values baked into the artifact. L4 self-subtract (α=−0.348) acts as a learned gate; L5 aggregates signal from L3+L4.
5. **LoRA-TTT improvements** (PR #1767) — Warm-start A matrix, alpha=144, WD=1.0. Phased score-first TTT (3 phases, 2000 prefix docs).

**Frozen alpha/beta converged values:**
```
beta  = [1.5973426, 1.8828125, 1.9921875]   # layers 3, 4, 5
alpha = [[0.2520, -0.0210, -0.0124],
         [0.0669, -0.3477,  0.0031],
         [0.1387,  0.2412,  0.0272]]
```

## Rule Compliance

- Score-first phased TTT (Condition 3), no pre-quant TTT, no n-gram cache
- Artifact ≤ 16 MB (max 15,976,882 B), train ≤ 600s, eval ≤ 600s
- Frozen recurrent scalars are trained weights serialized into the artifact — no parameters outside the 16 MB budget
- CaseOps legality pending issue #1604

## Test Plan

- Reviewer reproduces any single seed with the provided `train_gpt.py` and env vars from this README
- Verify artifact size `< 16,000,000` bytes in each seed log
- Verify score-first TTT ordering in code

🤖 Generated with Claude Code
