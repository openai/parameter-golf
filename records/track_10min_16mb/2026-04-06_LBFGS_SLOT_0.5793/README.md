# L-BFGS SLOT-8

**val_bpb: 0.5793** (3-seed mean, std 0.0009) | ~15.74 MB | 8xH100 SXM

## 3-Seed Results

| Seed | SLOT BPB | Eval Time | Artifact |
|------|----------|-----------|----------|
| 1337 | **0.5793** | 551s | 15,735,483 |
| 42 | **0.5784** | 543s | 15,730,615 |
| 2025 | **0.5801** | 543s | 15,746,295 |
| **Mean** | **0.5793** | | |

Beats #1313 (0.8637) by **0.284 BPP**. Beats merged SOTA (#1019, 1.1147) by **0.535 BPP**.

## Novel Technique: L-BFGS for SLOT Optimization

SLOT optimizes only **1,536 parameters** (512-dim delta + 1024-dim logit_bias) per sample. This is a tiny optimization problem — yet all previous submissions use first-order AdamW.

We replace AdamW with **L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno), a quasi-Newton method that uses curvature information via gradient history:

```python
slot_opt = torch.optim.LBFGS(
    [delta, logit_bias],
    lr=0.1,
    max_iter=5,           # 5 inner iterations per outer step
    history_size=10,      # 10 gradient pairs for Hessian approximation
    line_search_fn="strong_wolfe"  # optimal step size search
)
```

**Why L-BFGS is ideal here:**
- Second-order methods converge in O(n) steps for n-dimensional convex problems
- 1,536 params is tiny — L-BFGS memory overhead (history_size=10) is negligible
- Strong Wolfe line search guarantees sufficient decrease per step
- 8 outer steps × 5 inner = 40 effective function evaluations with curvature info
- AdamW needs 24-48 steps to achieve what L-BFGS does in 8

**Scaling results during development:**

| SLOT Method | BPB | Eval Time |
|-------------|-----|-----------|
| 24-step AdamW (PR #1313) | 0.8637 | ~300s |
| 24-step AdamW + hypergradient | 0.7625 | ~285s |
| 48-step AdamW + warm restart + hypergradient | 0.6321 | 453s |
| 12-step L-BFGS | 0.5765 | 809s (over budget) |
| **8-step L-BFGS** | **0.5793** | **551s** |

## Architecture (unchanged from #1313)

- 11L, 512d, 8 heads, 4 KV heads (GQA)
- LeakyReLU(0.5)² MLP with 3x expansion
- SmearGate + BigramHash + XSA-all + QK-Gain 4.0
- EMA + SWA + Late QAT + GPTQ int6 + lzma compression

## Compliance

- Score-first SLOT (frozen model, `torch.no_grad()` hidden states)
- No n-gram cache, no two-pass rescoring
- No eval-time training data access
- All seeds: train ≤600s, eval ≤600s, artifact ≤16MB
- Diagnostic evals (roundtrip, sliding window) skipped to maximize SLOT time budget

## Reproduction

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- Base architecture: PR #1303, PR #1313 (@anthony-maio)
- SLOT: Hu et al. arXiv:2505.12392v2, PR #1176 (@bigbag)
- L-BFGS: PyTorch built-in (Nocedal & Wright, Numerical Optimization)

Generated with [Claude Code](https://claude.com/claude-code)
