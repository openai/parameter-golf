# Hypergradient SLOT-24

**val_bpb: 0.7625** (3-seed mean, std 0.0027) | ~15.75 MB | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | + SLOT BPB | Steps | Artifact |
|------|------------|------------|-------|----------|
| 1337 | 1.1281 | **0.7654** | 5800 | 15,753,324 |
| 42 | 1.1273 | **0.7620** | 5798 | 15,774,360 |
| 2025 | 1.1271 | **0.7600** | 5793 | 15,734,660 |
| **Mean** | **1.1275** | **0.7625** | | |

Beats #1313 (0.8637) by **0.1012 BPB**. Beats merged SOTA (#1019, 1.1147) by **0.352 BPP**.

## Novel Technique: Hypergradient SLOT

Based on [arXiv:2502.11229](https://arxiv.org/abs/2502.11229) (Feb 2026): "Provable and Practical Online Learning Rate Adaptation with Hypergradient Descent."

Standard SLOT uses a fixed cosine LR schedule (0.012 → 0.001 over 24 steps). We replace this with **hypergradient descent** — the learning rate adapts itself each step:

```python
# Hypergradient: adapt LR based on gradient alignment
if step_i > 0:
    hg = sum((p.grad * prev_grad).sum() for p, prev_grad in zip(params, prev_grads))
    current_lr = clamp(current_lr + hyper_lr * hg, lr_min, lr_max)
```

**How it works:**
- Compute dot product between current gradient and previous gradient
- If positive (gradients consistent) → increase LR (converging, go faster)
- If negative (gradients flip) → decrease LR (overshooting, slow down)
- Auto-finds optimal stepsize per sample — no schedule tuning needed

**Why it helps:** Different documents have different optimization landscapes. A fixed cosine schedule is suboptimal — some samples converge fast (need high LR early), others need more careful steps. Hypergradient adapts per-sample.

**Overhead:** ~5 lines of code, negligible compute (one dot product per step).

## Architecture (unchanged from #1313 / PR #1303)

- 11L, 512d, 8 heads, 4 KV heads (GQA)
- LeakyReLU(0.5)² MLP with 3x expansion
- SmearGate + BigramHash + XSA-all + QK-Gain 4.0
- EMA + SWA + Late QAT + GPTQ int6 + lzma compression
- SLOT-24 with hypergradient descent (hyper_lr=1e-5)

## Compliance

- Score-first SLOT (frozen model, `torch.no_grad()` hidden states)
- No n-gram cache, no two-pass rescoring
- No eval-time training data access
- Self-contained, all seeds within time and size budgets
- Training: ~600s. Eval: ~350s. Total: ~16 min.

## Reproduction

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

No env vars needed beyond SEED.

## Credits

- Base architecture: PR #1303, PR #1313 (@anthony-maio)
- SLOT: Hu et al. arXiv:2505.12392v2, PR #1176 (@bigbag)
- Hypergradient descent: Baydin et al. arXiv:2502.11229
- Competition infrastructure: OpenAI, RunPod

Generated with [Claude Code](https://claude.com/claude-code)
