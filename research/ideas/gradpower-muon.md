# Idea — GradPower for Muon (port from #1682)

**Status:** 📝 CANDIDATE, bundled into spec 011.
**Source:** PR #1682 (PapaFranku4647, non-record).

## Core

One-line transform inside Muon's step, before momentum buffer + orthogonalization:

```python
g = torch.sign(g) * g.abs().pow(p)   # p = 0.9
```

Sign preserved; magnitudes softened for |g|>1, amplified for |g|<1.

## Evidence

#1682's matched 1×H100, 1200s, seed 1337 ablation:

| MUON_GRAD_POWER | val_bpb |
|---:|---:|
| 0.9 | 1.28338 |
| 1.0 | 1.28691 |

Δ = −0.00353 bpb. Local 4080 sweep over 3 seeds confirms same direction. Paper default p=1.2 LOSES in this setup.

## Why it might help on #1736

Muon orthogonalizes the update matrix via Newton–Schulz. If large-|g| entries dominate the pre-orthogonalization gradient, they disproportionately shape the orthogonal basis direction. Softening them (p<1) gives smaller entries more influence on the update direction — effectively "democratizing" which features get updated per step.

## Why it might not

- Author's regime is 1×H100 / 1200s. Ours is 8×H100 / 600s. Larger effective batch → cleaner gradients → the p=0.9 sweet spot could drift toward 1.0.
- #1736 already uses Muon momentum 0.95 + momentum warmup; the gradient reshape from p<1 may be partially redundant with momentum smoothing.

## Implementation sketch

In `Muon.step()` at line ~1593 of train_gpt.py:

```python
g = p.grad.bfloat16()
if h.muon_grad_power != 1.0:
    g = torch.sign(g) * g.abs().pow(h.muon_grad_power)
```

Env var `MUON_GRAD_POWER` (default 1.0 → identity / no-op, byte-compatible with #1736).

## Cross-references

- Companion lever: `research/ideas/per-layer-qk-gain.md`.
- Shelved standalone tapered-WD design: `research/specs/_shelved/011-tapered-wd.md`.
- Bundled into: `research/specs/011-training-bundle.md`.
