# Hybrid Spiking Neural Networks (SNNs) MLP

**val_bpb: 1.2982** | **15.78 MB** | 8×H100 SXM

A contest-friendly hybrid SNN submission built from the `train_gpt.py` baseline: keep dense GQA attention and the original training/eval/compression pipeline, but replace the standard  feed-forward block with a small multi-step leaky integrate-and-fire (LIF-style) spiking MLP.

Reference :https://arxiv.org/pdf/2203.14679

## Results (8×H100 80GB SXM)

| Run | Val loss | **Val bpb** | Serialized model | int8+zlib model | Total submission |
|-----|----------|-------------|------------------|-----------------|------------------|
| SNN baseline | 2.1919 | **1.2982** | 67,233,157 | 15,723,303 | **15,776,086** |

### Exact export log

- Serialized model: **67,233,157 bytes**
- Code size: **52,783 bytes**
- Total submission size: **67,285,940 bytes**
- Serialized model int8+zlib: **15,723,303 bytes**
  - payload: 17,179,020 bytes
  - raw_torch: 17,231,715 bytes
  - payload_ratio: 3.91x
- Total submission size int8+zlib: **15,776,086 bytes**
- final_int8_zlib_roundtrip val_loss: **2.19192126**
- final_int8_zlib_roundtrip val_bpb: **1.29817924**
- eval_time: **1435 ms**

## Key Innovation: Replace  MLP with a Multi-Step Spiking MLP

The baseline keeps the attention path dense and only swaps the MLP inside each Transformer block.

```python
# Standard baseline MLP
x = torch.relu(self.fc(x))
out = self.proj(x.square())
```

```python
# This submission: multi-step LIF-style spiking MLP
cur = self.fc(x)
mem = torch.zeros_like(cur)
spike_sum = torch.zeros_like(cur)
for _ in range(self.snn_steps):
    mem = decay * mem + cur / self.snn_steps
    over = mem - thresh
    spike_soft = torch.sigmoid(grad_scale * over)
    spike_hard = (over > 0).to(dtype=x.dtype)
    spike = spike_hard + spike_soft - spike_soft.detach()
    mem = mem - spike_hard * thresh
    spike_sum = spike_sum + spike
rate = spike_sum / self.snn_steps
out = self.proj(rate * self.spike_out_scale)
```

The spiking pathway introduces:
- **multi-step membrane integration** instead of a one-shot activation
- **thresholded firing** instead of continuous hidden activations
- **surrogate-gradient training** via a sigmoid straight-through estimator
- **spike-rate regularization** during training

This makes the FFN a small dynamical system rather than a static pointwise nonlinearity.

## Why this is interesting

This is **not** a fully spiking language model. It is a **hybrid Transformer + SNN-MLP** design:

- embeddings, attention, residual path, and logits remain standard dense LM components
- only the feed-forward block is replaced by a spiking mechanism
- the original Parameter Golf training and export path stays intact

That makes the experiment meaningful for the contest setting because it isolates one question:

> Can spike neural network achieves good performance in a tiny language model under a strict size budget?

## Training Architecture

Baseline model shape from `train_gpt.py`:

| Component | Setting |
|-----------|---------|
| Layers | 9 |
| Width | 512 |
| Attention | 8 heads, 4 KV heads (GQA) |
| Sequence length | 1024 |
| Vocab size | 1024 |
| MLP | **2× multi-step spiking MLP** |
| Embeddings | Tied |
| Position encoding | RoPE |
| Norm | RMSNorm |
| Residual structure | Encoder/decoder-style skip reuse |
| Logit stabilization | tanh softcap |
| Quantization/export | int8 + zlib |

### Spiking hyperparameters

| Parameter | Value |
|-----------|-------|
| `USE_SNN_MLP` | 1 |
| `SNN_STEPS` | 2 |
| `SNN_DECAY` | 0.8 |
| `SNN_THRESH_INIT` | 1.0 |
| `SNN_GRAD_SCALE` | 4.0 |
| `SNN_OUT_SCALE_INIT` | 1.0 |
| `SNN_RATE_LOSS` | 1e-4 |
| `SNN_RATE_TARGET` | 0.15 |

## Optimizer Setup

The script keeps the baseline split-optimizer recipe:

- **Adam** for token embeddings
- **Muon** for matrix-shaped parameters
- **Adam** for scalar/vector parameters
- optional tied-embedding head path from the baseline remains unchanged

This is important because the submission changes the model architecture without rewriting the overall training system.

## Run Command

```bash
RUN_ID=snn_baseline \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
USE_SNN_MLP=1 \
SNN_STEPS=2 \
SNN_DECAY=0.8 \
SNN_THRESH_INIT=1.0 \
SNN_GRAD_SCALE=4.0 \
SNN_OUT_SCALE_INIT=1.0 \
SNN_RATE_LOSS=1e-4 \
SNN_RATE_TARGET=0.15 \
VAL_LOSS_EVERY=1000 \
TRAIN_LOG_EVERY=200 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Practical Takeaway

This submission is best viewed as a **contest-friendly spiking experiment**, not as a claim that SNNs are already superior to standard dense LLMs on GPUs.

What it demonstrates:
- a hybrid SNN block can be dropped into the baseline with minimal code changes
- the resulting model still trains end-to-end with the standard PyTorch + Muon pipeline
- the exported artifact remains under the **16 MB** challenge limit after int8+zlib compression

## Limitations

- This version projects only the **average spike rate** back to model dimension, which is a fairly aggressive information bottleneck.
- It is likely more interesting as an architectural experiment than as the strongest contest-optimized submission.


## Credits

- **SNN adaptation**: `train_gpt_snn.py`, replacing the block MLP with a multi-step LIF-style spiking pathway while preserving the original contest pipeline
