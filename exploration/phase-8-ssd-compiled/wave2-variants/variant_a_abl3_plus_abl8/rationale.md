# Variant A: Ablation-3 (Bifurcated A_log) + Ablation-8 (Remove Autocast Disable)

## Why this combination

These are the #1 and #3 ablation wins from the iter-005.5 ablation sweep.

### Ablation-3: Bifurcated A_log initialization (val_bpb=1.873, delta=-0.119)

Replaces log-uniform A_log timescale initialization with a bifurcated scheme:
- 25% of heads initialized as "induction heads" (A_log ~ -4.0, giving A ~ -0.018, near-infinite memory)
- 75% of heads initialized as "local heads" (A_log in [0.3, 0.6], giving A in [-1.82, -1.35], fast decay)

This creates strong specialization pressure from step 0. Instead of the optimizer having to discover the right timescale distribution from a smooth log-uniform spread, it starts with two distinct populations that can immediately specialize: long-range copy/retrieval vs. local pattern matching. The largest single improvement in the sweep.

### Ablation-8: Remove autocast(enabled=False) in _prepare_ssd_inputs (val_bpb=1.915, delta=-0.077)

The baseline wraps `_prepare_ssd_inputs` in `torch.amp.autocast(enabled=False)`, forcing all SSD computations (einsums, reshapes, state operations) to run in fp32 even when the outer training loop has autocast enabled.

Removing this wrapper lets the einsums in `_ssd_chunk_pytorch` execute in bf16 under autocast, while cumsum operations (which need fp32 for numerical stability) are already handled by the cumsum→exp pipeline running in float explicitly. The result: faster SSD kernels with no measurable precision loss.

### Why they should stack

These modifications are **orthogonal**:
- Ablation-3 changes **initialization** (A_log parameter values at t=0)
- Ablation-8 changes **compute precision** (dtype of intermediate SSD tensors during training)

There is no mechanistic reason for interaction. Bifurcated A_log controls what the model learns; autocast removal controls how fast it computes. The only potential interaction: if bf16 einsums introduce noise that disrupts the bifurcated timescale structure, but the ablation-8 result (val_bpb=1.915, better than baseline's 1.992) shows bf16 SSD is stable.

## Expected outcome

If additive: 1.992 - 0.119 - 0.077 = **1.796** (theoretical floor)
Realistically: **1.83-1.87** (partial stacking with diminishing returns)

## Decision tree

```
IF val_bpb < 1.85:
    Both ablations stack additively. This is the submission candidate.
    Next step: run on 8xH100 for full 10-min budget, measure final BPB.
    If still < 1.85 at scale, prepare submission immediately.

ELIF val_bpb 1.85-1.90:
    Partial stacking. The autocast removal may interact with bifurcated init.
    Possible cause: bf16 noise slightly blurs the induction/local head boundary,
    reducing the specialization benefit of bifurcated A_log.
    Next step: still better than either ablation alone, so worth scaling up.
    Also try variant with explicit fp32 cumsum guard (keep autocast off only
    for the scan loop, not for _prepare_ssd_inputs).

ELIF val_bpb > 1.90:
    The ablations don't stack -- one masks the other's effect.
    Most likely: bf16 compute actively hurts when combined with the extreme
    A_log values (-4.0 for induction heads). The exp(-4.0) = 0.018 values
    may lose precision in bf16, negating the induction head advantage.
    Next step: revert to ablation-3 alone as the base and try other
    combinaitons (e.g., abl-3 + abl-2 vertical state carry removal).
```

## Exact changes from ablation-3 base

1. In `SSDMixer._prepare_ssd_inputs`: removed `with torch.amp.autocast(device_type=X.device.type, enabled=False):` wrapper. Body dedented by one level. This lets the PyTorch SSD path (einsums in `_ssd_chunk_pytorch`) run in bf16 under the outer autocast context, while the Triton path is unaffected (it manages its own dtypes explicitly).

2. Updated docstring to reflect the combined changes.

No hyperparameters were modified. No other code changes.
