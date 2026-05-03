# Notable Non-Record: Universal Transformer — 1.2249 BPB — Depth Recurrence with Iteration Embeddings

3 Unique Blocks × 4 Iterations = 12 Effective Layers + Per-Iteration Embeddings + 70% Param Savings + 4.95 MB Artifact

**val_bpb: 1.2249 (seed=42)** | 4.95 MB artifact | 8×H100 SXM, 555s training + 81s eval

## Results (seed=42, 8×H100 SXM)

| Metric | Value |
|--------|-------|
| val_bpb (post-quant sliding) | 1.2249 |
| val_bpb (pre-quant) | 1.2429 |
| val_loss | 2.0682 |
| Steps | 5,466 |
| ms/step | 101.55 |
| Training time | 555s |
| GPTQ time | 51s |
| Eval time | 81s |
| Peak memory | 26,212 MiB |
| Artifact | 4,946,680 bytes (4.95 MB) |
| Model bytes | 4,873,878 |
| Code bytes | 72,802 |
| Total params | 8,115,775 |
| Unique block params | 7,084,056 |
| Standard 11L params | 26,993,788 |
| **Param savings** | **70%** |

## What's Novel vs Existing Depth Recurrence (PR #363)

PR #363 (Evangeline Kamin) already explored depth recurrence extensively with 250+ hours of experiments. Our submission adds two features from the original Universal Transformer paper (Dehghani et al., 2018) that PR #363 did not implement:

1. **Per-iteration learnable embeddings** (timestep encoding): A learnable vector added to the hidden state before each block execution, telling the model which iteration it's on. Without this, all iterations are computationally identical — the model has no way to differentiate pass 1 from pass 4.

2. **Per-iteration learnable scales**: Modulate the residual contribution per effective layer, allowing different iterations to have different impact magnitudes.

### Comparison with PR #363

| | PR #363 (3x3) | Ours (3x4) |
|---|---|---|
| Unique blocks | 3 (stem) + 3 (core) + 3 (tail) = 9 | 3 shared across all |
| Iterations | 3 | 4 |
| Effective depth | 12 | 12 |
| Iteration embeddings | No | Yes |
| Iteration scales | No | Yes |
| Artifact | 15.6 MB | 4.95 MB |
| BPB | 1.1787 (sliding) | 1.2249 (sliding) |

Our BPB is worse because we have far fewer unique parameters (8.1M vs ~15M). But our artifact is 3x smaller, and the iteration embeddings are a principled addition from the paper.

## Architecture

- **3 unique transformer blocks** shared across 4 iterations
- Effective depth: 3 × 4 = 12 layers
- Block mapping: `effective_layer % 3` selects the shared block
- **Per-iteration learnable embeddings**: 12 vectors of dim 512, added to hidden state before each block
- **Per-iteration learnable scales**: 12 vectors of dim 512, modulate residual contribution
- d_model=512, 8 heads, 4 KV heads (GQA), 3x MLP with LeakyReLU(0.5)²
- U-Net skip connections adapted for looped structure
- VRL, SHC alpha/beta use effective layer count (12)
- Weight sharing verified: `block(0) is block(3) is block(6) is block(9)`
- int6 GPTQ (Hessian-aware, calibrated on training data within 600s budget)
- EMA(0.997), SWA, BigramHash(2048), SmearGate

## Command

```bash
USE_UNIVERSAL=1 \
UNIVERSAL_UNIQUE_BLOCKS=3 \
UNIVERSAL_ITERATIONS=4 \
NGRAM_EVAL=0 \
KNN_LAMBDA=0 \
SEED=42 \
OMP_NUM_THREADS=1 \
python3 -m torch.distributed.run --nproc_per_node=8 train_gpt.py
```

## Compliance

- [x] Artifact ≤16,000,000 bytes (4,946,680 — 31% of limit)
- [x] Training ≤600s on 8×H100 SXM (555s)
- [x] Eval ≤600s (81s)
- [x] GPTQ calibration inside training budget (51s, on training data)
- [x] No validation data during training
- [x] No network calls during evaluation
- [x] No external compute
- [x] No n-gram cache or kNN (clean sliding window eval only)
- [x] Reproducible from `train_gpt.py`

## References

- Universal Transformers: [arXiv:1807.03819](https://arxiv.org/abs/1807.03819) (Dehghani et al., 2018)
- Existing depth recurrence analysis: PR #363 by @evangelinehelsinki

## Included Files

- `train_gpt.py` — full training script
- `train_seed42.txt` — training log
- `submission.json` — metadata
- `run.sh` — reproduction script
- `requirements.txt` — dependencies
