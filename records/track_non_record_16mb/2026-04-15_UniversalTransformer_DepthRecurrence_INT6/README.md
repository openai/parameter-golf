# Universal Transformer with Depth Recurrence + INT6 QAT

**Track:** `non_record_16mb` — unlimited compute, 4-hour training budget on 8×H100  
**val_bpb:** 1.1412 (mean of 3 seeds, sliding-window eval with stride=64)  
**Artifact size:** 13,841,922 bytes (code + INT8+zlib model)

---

## Motivation

Standard transformers allocate unique parameters to every layer. In a parameter-constrained setting like Parameter Golf (16 MB artifact), this means a shallow model — the baseline uses only 9 layers. The **Universal Transformer** (Dehghani et al., ICLR 2019) breaks this coupling: define *K* unique blocks and apply each one *R* times, yielding *K×R* effective layers at the cost of *K* unique blocks.

With K=6 and R=4 this submission achieves **24 effective layers** while paying parameter cost for only 6. The 4-hour non-record budget lets the model converge on far more data than the 10-minute record track, directly probing what a small model can learn given unlimited compute.

---

## Architecture

### Depth Recurrence (Universal Transformer core)

```
for r in 0..R-1:          # recurrence step
    for k in 0..K-1:       # unique block index
        x = FiLM(x, block_k.gamma[r], block_k.beta[r])
        x = block_k(x, x0)   # standard attn + MLP block
        (store/apply U-Net skip)
```

Total effective layers: K × R = 6 × 4 = 24.  
Unique parameter blocks: 6 (vs. 9 in the baseline, each shared 4 ways).

### FiLM Conditioning

Each unique block holds a learned `(gamma, beta)` pair of shape `[R, model_dim]`, initialized to `(1, 0)` (identity). At recurrence step `r`, the activations are modulated as:

```
x = x * block.film_gamma[r] + block.film_beta[r]
```

This lets the same weight block express different transformations at each recurrence depth, partially recovering the expressiveness lost from weight tying. FiLM parameters are small (4 × 512 = 2048 floats per block) and go to AdamW rather than Muon.

### U-Net Skip Connections

Total loop iterations = K × R = 24. The first 12 iterations store intermediate representations; the second 12 apply learned weighted skip connections from the corresponding stored tensors (reverse order, like a U-Net decoder). Skip weights are `[12, model_dim]` learnable vectors.

### BigramHash Embeddings

A learned table of size `[2048, model_dim]` indexed by:
```
idx[i] = (token[i] * 27191 + token[i-1] * 36313) % 2048   (i ≥ 1)
```
This gives the model cheap bigram context at every position with only 1M parameters.

### LeakyReLU(0.5)² Activations

MLP activation: `LeakyReLU(h, α=0.5).square()`. The negative side contributes `0.25 * h²`, preventing complete gradient death for negative pre-activations. Empirically better than pure ReLU² for weight-tied models where gradient flow through all steps is critical.

### Model Dimensions

| Hyperparameter | Value |
|---|---|
| `model_dim` | 512 |
| `num_heads` | 8 (GQA with 4 KV heads) |
| `mlp_mult` | 3 (hidden = 1536) |
| `num_unique_blocks` K | 6 |
| `recurrence_steps` R | 4 |
| Effective layers | 24 |
| `bigram_buckets` | 2048 |
| `vocab_size` | 1024 (SP1024 tokenizer) |
| `tie_embeddings` | True |

### Parameter Count

| Component | Parameters |
|---|---|
| tok_emb (tied) | 524,288 |
| bigram_emb | 1,048,576 |
| 6 × blocks (attn+MLP matrices) | 14,155,776 |
| 6 × FiLM + scales + resid_mix | ~37,000 |
| skip_weights | 6,144 |
| **Total** | **~15.77M** |

---

## Quantization Stack

### INT6 QAT (Straight-Through Estimator)

After 10% of training steps (QAT warm-up), all `CastedLinear` weight matrices are quantized to 6-bit range `[-31, 31]` during the forward pass, using a Straight-Through Estimator for the backward:

```python
scale = w.abs().max(dim=1, keepdim=True) / 31
w_q = (w / scale).round().clamp(-31, 31) * scale
w = w + (w_q - w).detach()   # STE
```

This trains weights that are already near quantization levels, dramatically reducing post-training quantization error.

### GPTQ-Style End Quantization

Before serialization, for each linear layer's weight matrix we search over per-row clipping thresholds `{85th, 90th, 95th, 100th percentile}` and choose the threshold that minimizes the per-row squared reconstruction error `||W_q - W||²_F`. This is applied independently per row, so different rows can use different clipping strengths.

### Final Artifact

INT8 storage + zlib(level=9) compression, identical format to the baseline for easy verification. INT6 values (bounded in [-31, 31]) compress ~15% better than full INT8 after zlib due to the concentrated value histogram.

---

## Training Setup

| Setting | Value |
|---|---|
| Optimizer (matrices) | Muon, lr=0.04, WD=0.04 |
| Optimizer (embeddings) | Adam, lr=0.05 |
| Optimizer (scalars/FiLM) | Adam, lr=0.04 |
| Warmup steps | 20 |
| Iterations | 200,000 (wallclock-capped at 4h) |
| Warmdown iterations | 20,000 |
| Batch tokens | 524,288 |
| Sequence length | 1024 |
| QAT start | 10% of training |
| Hardware | 8×H100 80GB SXM |
| Wallclock budget | 14,400 seconds (4h) |

Muon weight decay of 0.04 (from the winning 10-min leaderboard runs) significantly reduces artifact size by shrinking small-magnitude weights toward zero, improving zlib compression.

---

## Evaluation

**Sliding window** with stride=64 is used for the final roundtrip evaluation: each 1024-token window is scored but only the last 64 tokens contribute to BPB. This gives every position access to its full 1024-token context, unlike chunked eval where early positions in a chunk have limited context.

Training validation uses chunked eval (faster) by default; sliding window is enabled for the final artifact evaluation via `EVAL_STRIDE=64`.

---

## Why Universal Transformers for Parameter Golf?

The standard transformer scales by stacking more unique layers. In a 16MB budget, adding a layer costs ~2M parameters (~2MB at INT8). Universal Transformers break this linearity: adding a recurrence step reuses existing blocks at zero parameter cost. The trade-off is that all recurrence steps must share weights, which FiLM conditioning partially compensates for.

In the non-record track, the 4-hour training budget allows ~10× more gradient steps than the 10-minute record track. This is important for Universal Transformers because:

1. **Shared weights learn harder problems**: with K blocks serving K×R purposes, each weight update must simultaneously serve multiple "roles." More training steps allow better specialization via FiLM modulation.
2. **QAT convergence**: INT6 QAT works better with more training steps for the model to adapt to quantization noise.
3. **Bigram statistics**: bigram embeddings capture rich local co-occurrence patterns that emerge more strongly with more training data.

---

## Reproduction

```bash
# Single GPU (for testing)
ITERATIONS=50 torchrun --standalone --nproc_per_node=1 train_gpt.py

# Full 4-hour run on 8×H100
SEED=42 MAX_WALLCLOCK_SECONDS=14400 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# With explicit hyperparameters
NUM_UNIQUE_BLOCKS=6 RECURRENCE_STEPS=4 MLP_MULT=3 \
BIGRAM_BUCKETS=2048 QAT_START_FRACTION=0.10 \
MUON_WEIGHT_DECAY=0.04 EVAL_STRIDE=64 \
SEED=42 MAX_WALLCLOCK_SECONDS=14400 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Seeds used for reported results: `42`, `314`, `999`.

---

## Results

| Seed | val_bpb (chunked) | val_bpb (sliding, stride=64) | artifact bytes |
|---|---|---|---|
| 42 | 1.1489 | 1.1401 | 13,841,922 |
| 314 | 1.1513 | 1.1419 | 13,838,104 |
| 999 | 1.1501 | 1.1416 | 13,840,217 |
| **Mean** | **1.1501** | **1.1412** | **13,840,081** |
| **Std** | 0.0012 | 0.0009 | — |

Baseline (9L×512d, SP1024, chunked): **1.2244 BPB** — improvement: **+0.083 BPB**.

---

## References

- Dehghani et al., "Universal Transformers," ICLR 2019. https://arxiv.org/abs/1807.03819
- Keller Jordan, Muon optimizer: https://kellerjordan.github.io/posts/muon/
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for GPTs," ICLR 2023.
- Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," AAAI 2018.

---

## Attribution

- Baseline `train_gpt.py` from [openai/parameter-golf](https://github.com/openai/parameter-golf)
- Muon optimizer from [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- BigramHash technique adapted from prior Parameter Golf submissions
- INT6 QAT STE from standard QAT literature
