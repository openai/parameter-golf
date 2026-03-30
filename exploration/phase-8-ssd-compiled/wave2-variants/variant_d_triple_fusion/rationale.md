# Variant D: Triple Fusion

## Hypothesis

If the three winning ablations are independent (i.e., they improve BPB through orthogonal mechanisms), their effects should stack additively.

Individual ablation results vs baseline (val_bpb=1.966):
- **Ablation-3** (bifurcated A_log): val_bpb=1.873 (delta = -0.093)
- **Ablation-2** (no vertical state carry): val_bpb=1.910 (delta = -0.056)
- **Ablation-8** (no autocast(enabled=False)): val_bpb=1.915 (delta = -0.051)

If independent: 1.966 - 0.093 - 0.056 - 0.051 = **1.766** (optimistic bound)

More conservatively, starting from the ablation-3 champion: 1.873 - 0.056 - 0.051 = **~1.77**

## Changes Applied

### 1. Bifurcated A_log initialization (from ablation-3 -- already in base)

25% of heads initialized as "induction heads" (A_log ~ -4.0, near-infinite memory for copy/retrieval patterns). 75% as "local heads" (A_log in 0.3-0.6, fast decay for bigram/phrase patterns). This creates strong specialization pressure compared to log-uniform which spreads heads across all timescales without bias.

**Mechanism**: Forces the optimizer to specialize heads early rather than discovering the induction/local split from scratch. The bifurcation matches the bimodal distribution that trained models converge to anyway.

### 2. Disable vertical state carry (from ablation-2)

Set `vertical_states=None` in the depth recurrence loop. The vertical carry passes per-chunk SSM states from iteration i to iteration i+1, creating a 2D recurrence grid (sequence x depth). Removing it simplifies to pure horizontal (sequence-only) state carry at each depth iteration.

**Mechanism**: The vertical carry adds O(nchunks * nheads * d_state * headdim) memory operations per iteration. With only 4 iterations, the representational benefit may not justify the optimization difficulty (the optimizer must learn useful 2D state interactions). With bifurcated A_log, the heads are already specialized -- vertical carry may just add noise to the specialization.

### 3. Remove autocast(enabled=False) in _prepare_ssd_inputs (from ablation-8)

The `_prepare_ssd_inputs` method had `torch.amp.autocast(device_type=..., enabled=False)` which forced all tensor reshaping, validation, and state initialization to run in fp32. Removing it lets the autocast context from the caller propagate through, allowing bf16 operations where appropriate.

**Mechanism**: The autocast(enabled=False) was a conservative correctness guard -- ensuring no precision loss during state setup. But the actual compute (cumsum, exp, einsum) in `_ssd_chunk_pytorch` already handles precision explicitly. The guard just adds overhead by forcing unnecessary fp32 casts on reshape/contiguous calls.

## Independence Argument

These three ablations target different subsystems:
- **A_log init**: Affects parameter initialization only (one-time effect at step 0)
- **Vertical carry**: Affects forward pass data flow (structural change to computation graph)
- **Autocast guard**: Affects numerical precision policy (dtype casting behavior)

No obvious coupling. The strongest potential interaction is between vertical carry and A_log: if bifurcated heads rely on vertical state transfer for induction behavior, removing vertical carry would negate the bifurcation benefit. But ablation-2 (which used log-uniform A_log + no vertical carry) still improved over baseline, suggesting vertical carry hurts regardless of A_log init.

## Decision Tree

```
IF val_bpb < 1.80:
    All three stack. This is likely the submission candidate.
    Next: optimize hyperparameters (LR, n_iters, d_state) around this config.

ELIF val_bpb in [1.80, 1.87]:
    Partial stacking. Two of three interact.
    Next: run pairwise combinations to identify which pair conflicts.
    - variant_d1: ablation-3 + ablation-2 (bifurcated + no vertical)
    - variant_d2: ablation-3 + ablation-8 (bifurcated + no autocast guard)
    - variant_d3: ablation-2 + ablation-8 (no vertical + no autocast guard)

ELIF val_bpb > 1.87:
    Ablations NOT independent -- they share a common cause.
    The baseline had a single underlying problem, and each ablation partially fixed it.
    Next: investigate what the common root cause might be (likely precision-related
    given autocast is involved).
```

## Run Command

```bash
# 1xH100 smoke test
RUN_ID=variant_d_triple_fusion torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 full run
RUN_ID=variant_d_triple_fusion_full torchrun --standalone --nproc_per_node=8 train_gpt.py
```
