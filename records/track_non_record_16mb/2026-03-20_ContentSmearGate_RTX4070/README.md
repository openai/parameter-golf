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

### content_scale initialization sweep (200 steps, standard eval)

| content_scale init | val_bpb | vs Static |
|---|---|---|
| **0.1** | **1.8962** | **-0.0825** |
| 0.3 | 1.9133 | -0.0654 |
| 0.5 | 1.9346 | -0.0441 |
| Static (control) | 1.9787 | baseline |
| 1.0 | 1.9910 | +0.0123 |
| 2.0 | 2.1425 | +0.1638 |

**Key finding: the content signal should be subtle (scale=0.1), not dominant.** Too much content modulation (≥1.0) hurts — the gate becomes too reactive and destabilizes early training. A light touch lets the model learn when blending helps without overwhelming the base gate.

### Full run (500 steps, standard eval, content_scale=0.5)

| Variant | val_bpb | ms/step | Artifact |
|---------|---------|---------|----------|
| Static SmearGate (SOTA arch) | 1.7080 | 1162 | 15.57MB |
| **Content SmearGate** | **1.6795** | **958** | 15.80MB |
| Delta | **-0.0285** | **-17.5%** | +0.23MB |

Note: the 500-step run used content_scale=0.5 (before the sweep). The optimal init of 0.1 found in the sweep should yield further improvement at 500 steps.

## Run Command

```bash
SEED=42 \
CONTENT_SCALE_INIT=0.1 \
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

- Full 500-step run with content_scale=0.1 (expected to beat 1.6795)
- Validate on 8xH100 at full 20K steps (compute grant pending)
- Explore trigram hash and multi-step lookback extensions
