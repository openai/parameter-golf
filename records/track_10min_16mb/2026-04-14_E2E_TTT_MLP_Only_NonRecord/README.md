# E2E TTT (MLP-Only, Last 1/2 Blocks) — Non-Record Idea Submission

**val_bpb: 1.1104** (seed 1337, single-seed) | **13.85 MB** | 8×H100 SXM

Non-record idea submission demonstrating a paper-aligned implementation of
**End-to-End Test-Time Training** (arXiv:2512.23675, ICLR 2026) on top of
PR #1493's merged SOTA stack, targeting OpenAI's "E2E TTT" wishlist item.

## Why non-record

1. **SP1024, not SP8192.** The SP8192 data used by PR #1493 is not available
   from the `willdepueoai/parameter-golf` HF repo (only SP1024, SP4096 via other
   means). We ran on SP1024, which gives ~0.03 BPB penalty vs SP8192 baselines.
2. **Single seed** (1337). Three-seed validation would be standard for a record
   submission.
3. **TTT gain at 27M scale is marginal.** All three ablations landed in 1.110–
   1.112 BPB — sliding window alone carries almost all the eval-time gain
   (−0.024 BPB) while TTT adds only −0.001 BPB. The paper's technique was
   validated on 3B+ models; we're demonstrating diminishing returns at 27M.

## Technique

End-to-End TTT (paper section 3) updates MLP weights at eval time via
score-first SGD on already-scored chunks. We freeze everything except MLP
layers in the last N blocks:

```
for chunk in validation_tokens:
    with torch.no_grad():
        logits = forward(chunk)
        NLL = cross_entropy(logits, chunk_targets)
        commit_NLL_to_running_total()           # <<< score-before-update
    optimizer.zero_grad()
    loss = forward_with_grad(chunk).mean_NLL
    loss.backward()
    optimizer.step()                             # updates only MLP weights
                                                 # in blocks [start_block:]
```

The paper's key finding is that updating embeddings, attention, or norms
during TTT causes instability — only the MLP layers are safe to update.
We implement this with a `TTT_E2E_MODE=1` flag that filters `ttt_params`
in `eval_val_ttt`:

```python
if h.ttt_e2e_mode:
    ttt_start_block = max(0, int(h.num_layers * (1 - h.ttt_e2e_last_frac)))
    ttt_params = []
    for name, p in base_model.named_parameters():
        in_block = name.startswith('blocks.')
        is_mlp = '.mlp.' in name
        idx = int(name.split('.')[1]) if in_block else -1
        p.requires_grad_(is_mlp and idx >= ttt_start_block)
        if p.requires_grad:
            ttt_params.append(p)
```

## Results

**Base model** (shared across all TTT configs):
- 11L × 512d, GQA 8/4 heads, SP1024, 3-layer depth recurrence (L3-5),
  parallel residuals (L7+), QK-Gain 5.0
- Training: 3,940 steps in 588s (wallclock cap), final val_bpb 1.1250
- GPTQ SDClip int6 + brotli → 13.85 MB artifact

**TTT ablations** (all seed 1337, config differs only in TTT params):

| Run | TTT_LR | Epochs | LAST_FRAC | Trainable Params | val_bpb | Eval Time |
|-----|-------:|-------:|----------:|-----------------:|--------:|----------:|
| #1 defaults | 0.005 | 3 | 0.25 | 6.3M | 1.11137 | 408s |
| #2 broader | **0.015** | **2** | **0.50** | **12.6M** | **1.11037** | 443s |
| #3 aggressive | 0.05 | 5 | 0.09 | 2.1M | 1.11112 | 413s |

The **best config (run #2)** updates the MLP layers of blocks 5–10 with
moderate LR and 2 epochs per chunk. Run #3 (aggressive single-block) and
run #1 (paper defaults) both underperformed — suggesting the sweet spot
is a moderate-breadth, moderate-LR configuration.

## Gain decomposition

| Eval stage | BPB | Δ vs prev |
|------------|----:|----------:|
| Post-quantization | 1.1360 | — |
| + Sliding window (stride 64) | 1.1123 | **−0.0237** (big win) |
| + E2E TTT (run #2) | 1.1104 | −0.0019 (marginal) |

The sliding-window eval is doing ~95% of the eval-time improvement.
E2E TTT contributes a real but small additional gain at 27M scale.

## Compliance (Issue #1017)

All four conditions satisfied:

1. **Causal dependence** — Each chunk is scored under `torch.no_grad()` using
   only the prefix that has been committed. The TTT update uses only that
   same chunk's loss, computed after the score was committed.
2. **Normalized distribution** — Standard softmax over the full SP1024 vocab.
   No n-gram mixing, no hash probabilities.
3. **Score-before-update** — The explicit `with torch.no_grad(): ...` block
   around the forward pass and NLL accumulation happens BEFORE the
   `optimizer.zero_grad()`/`.backward()`/`.step()` sequence for that chunk.
4. **Single left-to-right pass** — Chunks are processed sequentially in order;
   each token is scored exactly once; no rescoring.

## Takeaway for future submissions

If you want to use E2E TTT as a lever to improve BPB, either (a) scale up
the model significantly, or (b) increase training budget so the base model
is further from its ceiling. At 27M params × 588s training, there's simply
not enough adaptation room left for test-time updates to meaningfully help.
