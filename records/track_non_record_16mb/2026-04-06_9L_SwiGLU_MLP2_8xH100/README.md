# Non-Record Submission: 9L SwiGLU Activation (8×H100)

This submission documents replacing the baseline **ReLU²** MLP activation with **SwiGLU**
(as used in Llama 2/3, PaLM, and Mistral), trained on a full 8×H100 run under the 10-minute
wallclock cap.

**Final post-quant score: 1.23697794 val_bpb** — competitive with the 8×H100 naive baseline
(1.2244) and a meaningful improvement over the stock single-GPU baseline.

## What Changed

### SwiGLU Activation (replacing ReLU²)

The baseline MLP computes:
```python
# relu^2
x = relu(fc(x))
return proj(x * x)
```

SwiGLU uses two input projections and a gating mechanism:
```python
# SwiGLU — as in Llama/PaLM
return proj(silu(gate(x)) * up(x))
```

This allows the network to **selectively suppress irrelevant features** via the SiLU gate,
yielding better signal-to-noise per parameter. The hidden dimension is scaled to `2/3 × mlp_mult × dim`
so total parameter count remains equal to the ReLU² baseline at the same `MLP_MULT`.

**Paper:** [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)

### GQA Compatibility Fix

The baseline `F.scaled_dot_product_attention(enable_gqa=...)` argument is not available in
older PyTorch versions. Replaced with explicit `repeat_interleave` expansion of K/V heads:

```python
if self.num_kv_heads != self.num_heads:
    r = self.num_heads // self.num_kv_heads
    k = k.repeat_interleave(r, dim=1)
    v = v.repeat_interleave(r, dim=1)
y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
```

## Configuration

```
NUM_LAYERS=9   MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_MULT=2     USE_SWIGLU=1
TRAIN_BATCH_TOKENS=524288  TRAIN_SEQ_LEN=1024
MAX_WALLCLOCK_SECONDS=600
GPU: 8×H100 SXM
```

## Results

| Metric | Value |
|---|---|
| `val_bpb` (post-quant roundtrip) | **1.23697794** |
| `val_bpb` (pre-quant, at stop) | 1.2302 |
| Artifact size (int8+zlib) | **15,900,595 bytes** |
| Model size (int8+zlib) | 15,852,180 bytes |
| Code size | 48,415 bytes |
| Steps completed | 12,075 / 20,000 |
| Wallclock | 600s |
| Peak GPU memory | 10,725 MiB |
| GPU | 8×H100 SXM |

## Experiment Journey (Mac → GPU)

All Mac experiments used `train_gpt_mlx.py` with 500-step runs as fast architecture tests.

| Exp | Config | val_bpb | Size | Valid |
|---|---|---|---|---|
| Smoke | 9L relu² MLP2 (baseline) | 2.3555 | 10.6 MB | ✅ |
| exp1 | 10L relu² MLP2 | 2.0777 | 12.5 MB | ✅ |
| exp3 | 12L SwiGLU MLP3 | 1.9986 | 17.5 MB | ❌ over |
| exp5 | 10L SwiGLU MLP3 | 2.0299 | 15.4 MB | ✅ |
| **GPU final** | **9L SwiGLU MLP2 (8×H100)** | **1.2370** | **15.9 MB** | **✅** |

## Files

- `train_gpt.py` — training script snapshot with SwiGLU and GQA fix applied
- `submission.json` — leaderboard metadata
