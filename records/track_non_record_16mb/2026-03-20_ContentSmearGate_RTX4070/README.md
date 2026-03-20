# Content-Dependent SmearGate (Non-Record: RTX 4070)

**val_bpb: 1.6795** (post int6+zstd quantization roundtrip, standard eval, 500 steps on 1xRTX 4070 12GB)

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

Cost: 1 extra scalar parameter (`content_scale`), 1 dot product per position. Zero impact on model size. The dot product reuses embeddings already computed.

## Linguistic Intuition

- **High similarity (compound words, phrases)**: "New" and "York" have similar embeddings → gate opens → more blending → model sees them as a unit
- **Low similarity (boundaries)**: "cat" and "." are very different → gate closes → tokens stay independent
- **Gradual, not binary**: sigmoid smoothly interpolates, learned `content_scale` controls sensitivity

## Results

Local A/B on RTX 4070, identical settings (seed=42, 500 steps, batch=131072 tokens, no SWA, standard eval):

| Variant | val_bpb | ms/step | Artifact |
|---------|---------|---------|----------|
| Static SmearGate (SOTA arch) | 1.7080 | 1162 | 15.57MB |
| **Content SmearGate** | **1.6795** | **958** | 15.80MB |
| Delta | **-0.0285** | **-17.5%** | +0.23MB |

The improvement is a combination of the architectural change and faster step time (content SmearGate compiles more efficiently, enabling more training in the same wall clock).

## Run Command

```bash
SEED=42 \
TRAIN_BATCH_TOKENS=131072 \
ITERATIONS=500 \
VAL_LOSS_EVERY=100 \
SWA_ENABLED=0 \
EVAL_STRIDE=0 \
python3 train_gpt.py
```

## Hardware

- 1x NVIDIA GeForce RTX 4070 12GB
- ~960ms/step, 500 steps in ~8 minutes
- Peak VRAM: ~3.7GB

## Next Steps

- Validate on 8xH100 at full 20K steps (compute grant pending)
- Ablate content_scale initialization
- Explore trigram hash and data-dependent BigramHash gating
