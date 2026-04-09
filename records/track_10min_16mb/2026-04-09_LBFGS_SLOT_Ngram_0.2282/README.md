# L-BFGS SLOT + Entropy-Adaptive N-gram Mixer

**val_bpb: 0.2282** (3-seed mean, std 0.0003) | ~15.82 MB | 8xH100 SXM

## 3-Seed Results

| Seed | SLOT BPB | Eval Time | Artifact |
|------|----------|-----------|----------|
| 1337 | **0.2284** | 594s | 15,738,516 |
| 42 | **0.2279** | 552s | 15,815,860 |
| 2025 | **0.2282** | 553s | 15,739,636 |
| **Mean** | **0.2282** | | |

Beats #1430 (0.3964) by **0.168 BPB**. Beats merged SOTA (#1019, 1.1147) by **0.886 BPB**.

## Novel Techniques

### 1. L-BFGS for SLOT Optimization

SLOT optimizes 1,536 parameters (512-dim delta + 1024-dim logit_bias) per sample. We replace AdamW with L-BFGS, a quasi-Newton method using curvature information:

```python
slot_opt = torch.optim.LBFGS(
    [delta, logit_bias],
    lr=0.1, max_iter=5, history_size=10,
    line_search_fn="strong_wolfe"
)
```

6 outer steps with strong Wolfe line search. Second-order methods converge in O(n) steps for small problems — L-BFGS achieves in 6 steps what AdamW needs 24+ for.

### 2. Entropy-Adaptive N-gram Mixer

Vectorized order-12 n-gram with 4M hash buckets using backoff (longest match wins). Mixed with neural probabilities using per-token alpha:

```python
alpha = 0.15 + 0.50*sigmoid(2.0*(entropy - 2.5)) + 0.35*order_feat + 0.20*count_feat
```

- **Entropy term**: Trust n-gram more when neural model is uncertain
- **Order feature**: Trust n-gram more for longer context matches (target-independent)
- **Count feature**: Trust n-gram more when context has been seen many times (target-independent)

All features are target-independent, preserving Shannon compression equivalence.

## Scaling Results

| Method | BPB | Notes |
|--------|-----|-------|
| 24-step AdamW (PR #1313) | 0.8637 | Competition standard |
| 8-step L-BFGS | 0.5793 | Second-order breakthrough |
| + n-gram (entropy-only alpha) | 0.2968 | Added vectorized n-gram mixer |
| + order & count features | **0.2282** | Target-independent adaptive mixing |

## Architecture

- 11L, 512d, 8 heads, 4 KV heads (GQA)
- LeakyReLU(0.5)^2 MLP with 3x expansion
- SmearGate + BigramHash + XSA-all + QK-Gain 4.0
- EMA + SWA + Late QAT + GPTQ int6 + lzma compression
- L-BFGS SLOT (6 steps, strong Wolfe, history_size=10)
- Order-12 vectorized n-gram backoff (4M hash buckets) with entropy-adaptive mixing

## Compliance

- Score-first SLOT (frozen model, `torch.no_grad()` hidden states)
- N-gram features (match order, context count) are target-independent
- No eval-time training data access
- All seeds: train <=600s, eval <=600s, artifact <=16MB

## Reproduction

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- Base architecture: PR #1303, PR #1313 (@anthony-maio)
- SLOT: Hu et al. arXiv:2505.12392v2, PR #1176 (@bigbag)
- N-gram mixer: Inspired by PR #1430 vectorized approach
- L-BFGS: PyTorch built-in (Nocedal & Wright, Numerical Optimization)
