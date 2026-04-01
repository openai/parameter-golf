# Content-Dependent SmearGate (Non-Record: Negative Result)

## Motivation

The current SmearGate uses a **static, learned gate** per dimension to blend each token embedding with its predecessor. The gate values are fixed regardless of what the tokens actually are — "New York" gets the same blending ratio as "cat .".

This submission replaces static SmearGate with **content-dependent SmearGate**: the gate is modulated by the dot-product similarity between adjacent token embeddings. Similar tokens blend more, dissimilar tokens blend less.

## The Change

Original SmearGate:
```python
g = sigmoid(self.gate)  # fixed per dimension
output = (1 - g) * x + g * x_prev
```

Content-dependent SmearGate:
```python
similarity = dot(x, x_prev) / sqrt(dim)  # how similar are adjacent tokens?
g = sigmoid(self.gate + content_scale * similarity)  # modulate gate by similarity
output = (1 - g) * x + g * x_prev
```

Cost: 1 extra scalar parameter (`content_scale`), 1 dot product per position.

## Linguistic Intuition

- **High similarity (compound words, phrases)**: "New" and "York" have similar embeddings → gate opens → more blending → model sees them as a unit
- **Low similarity (boundaries)**: "cat" and "." are very different → gate closes → tokens stay independent
- **Gradual, not binary**: sigmoid smoothly interpolates, learned `content_scale` controls sensitivity

## H100 SXM Results (1xH100, 10-min wallclock)

| Variant | Steps | val_bpb (post int6+zstd) | ms/step | Eval |
|---------|-------|--------------------------|---------|------|
| Static SmearGate | 869 | **1.5209** | 691 | sliding stride=64 |
| Content SmearGate (scale=0.1) | 803 | 1.5524 | 748 | standard |

**Negative result.** The content dot-product adds ~8% per-step overhead, reducing total training steps from 869 to 803 in the fixed 10-minute window. The quality improvement from content-dependent gating does not compensate for the lost training steps.

Note: eval methods differ (sliding vs standard), which accounts for ~0.02-0.03 BPB in favor of static. Even adjusting for this, content SmearGate is at best a wash — and it trains fewer steps.

## Why It Doesn't Work (Analysis)

In the parameter golf setting, **wall-clock efficiency dominates architecture quality**. The 10-minute training cap means every millisecond of per-step overhead directly costs training iterations. The content dot-product:

1. Adds a `(x * x_prev).sum(dim=-1)` per position — cheap in isolation
2. But breaks torch.compile optimization patterns, adding ~57ms/step on H100
3. Over 10 minutes, this costs 66 training steps (869 → 803)
4. The content-dependent gating would need to improve per-step learning by >8% to break even — it doesn't

**The static gate works well enough** because the transformer's self-attention layers already learn token-pair relationships. SmearGate only needs to provide a rough prior; making it content-dependent adds complexity where the model doesn't need it.

## Potential Fixes (Untested)

- **Pre-compute similarity at embedding time** and cache it, avoiding recomputation in the compiled forward pass
- **Use a cheaper proxy** than full dot-product (e.g., XOR hash of token IDs mapped to a small learned table)
- **Only apply content gating to a subset of dimensions** to reduce compute while keeping some benefit

## Run Command

```bash
# On RunPod 1xH100 SXM:
SMEAR_MODE=content \
CONTENT_SCALE_INIT=0.1 \
RUN_ID=content_smeargate \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
python3 train_gpt.py
```

## Hardware

- H100 SXM 80GB (RunPod, 1 GPU)
- Static: 691ms/step, 869 steps in 10 min
- Content: 748ms/step, 803 steps in 10 min
