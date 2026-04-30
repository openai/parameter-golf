# SmearGate + OrthoInit + Muon WD + Int6 STE QAT + MLP 3x + Sliding Window

**val\_bpb: 1.1556** (post-quant int6+zstd-22, sliding window eval stride=64)

## Summary

A 22.4M parameter transformer language model trained in under 10 minutes on 8×H100 GPUs, compressed to a 15.1MB artifact via int6 quantization-aware training and zstd-22. The architecture combines a SmearGate bigram embedding layer, orthogonal weight initialization, 3× MLP expansion, U-Net skip connections, and decoupled Muon weight decay, evaluated with sliding window context at stride 64.

## Architecture

### Transformer Core

A 9-layer, 512-dim transformer with 8 attention heads (4 KV heads via grouped-query attention) and tied input/output embeddings over a 1024-token BPE vocabulary. Sequence length during training is 1024 tokens.

### SmearGate

A learned per-dimension gate (\~512 params) that blends each token's embedding with the previous token's embedding before the transformer processes anything:

```python
gate = sigmoid(self.gate)  # shape \[dim], init ≈ 0.95
output = gate \* current\_emb + (1 - gate) \* prev\_token\_emb
```

This injects bigram (two-token) context directly into the embedding layer. Normally a transformer must discover token-pair relationships through self-attention; SmearGate provides this signal for free. The gate is initialized via `sigmoid(3.0) ≈ 0.95` so it starts near-identity (mostly current token), and the model learns per-dimension how much previous-token blending is useful.

Applied after embedding lookup and bigram hash addition, before RMS normalization.

### Bigram Hash Embedding

A 4096-bucket hash table (dim=128, projected to 512) maps consecutive token pairs to learned embeddings via `(prev \* 92821 + cur) % 4096`. This gives the model direct access to token-pair features at minimal parameter cost.

### MLP 3× Expansion

MLP hidden dimension is 3× the model dimension (1536 for a 512-dim model). The space savings from int6 quantization fund this extra capacity — wider MLPs allow more expressive nonlinear feature transformation between attention operations.

### U-Net Skip Connections

The 9-layer transformer is split into an encoder half (4 layers) and a decoder half (5 layers) with learned skip weights connecting corresponding encoder/decoder layers. This gives the decoder direct access to earlier representations without relying solely on the residual stream.

## Training

### Muon Optimizer with Weight Decay

The Muon optimizer (MomentUm Orthogonalized by Newton-Schulz) runs SGD with Nesterov momentum, then post-processes each 2D parameter's gradient update by replacing it with the nearest orthogonal matrix via 5-step Newton-Schulz iteration. This is equivalent to steepest descent under the spectral norm, improving the conditioning of the optimization landscape.

Decoupled weight decay (`p.mul\_(1 - wd \* lr)`, wd=0.01) is applied before each gradient update. This keeps weights smaller and better-distributed, which directly benefits both generalization and downstream quantization — tighter weight distributions quantize into fewer int6 buckets with less error and compress better with zstd.

Momentum is warmed from 0.92 → 0.99 over the first 1500 steps.

### Orthogonal Weight Initialization

All non-zero-init CastedLinear weight matrices are initialized with `nn.init.orthogonal\_()`. Orthogonal matrices have all singular values equal to 1, meaning gradients flow uniformly through the network at initialization with no vanishing or exploding signals. Additionally, since Muon's Newton-Schulz step orthogonalizes updates, starting from an already-orthogonal matrix means early updates are immediately useful rather than spent correcting a random initialization. With only \~12k steps in the 10-minute budget, faster convergence matters.

### Int6 Quantization-Aware Training (STE)

All 2D weight matrices are fake-quantized to int6 (\[-31, 31]) during every forward pass via Straight-Through Estimator — the forward pass sees quantized weights while gradients flow through the rounding operation as if it were identity. The model learns weight configurations that are inherently robust to post-training quantization. The tied embedding matrix is stored as fp16 passthrough (not quantized), since it serves double duty for both input embeddings and output predictions where errors compound in both directions.

### Learning Rate Schedule

Warmup over 20 steps, followed by linear warmdown over the final 3000 steps. Separate learning rates for tied embeddings (0.030), matrix parameters (0.020), and scalar parameters (0.020).

## Evaluation

### Sliding Window (stride=64)

Instead of chopping validation text into non-overlapping chunks (where tokens near the start of each chunk lack context), sliding window uses overlapping windows with stride 64 and the full 1024-token context window. Each scored token gets 960+ tokens of prior context. This is purely an evaluation-time technique — it does not change the model.

## Export

### Int6 + zstd-22 Compression

All quantized weights are packed into int8 containers and compressed with zstandard at level 22. The int6 representation plus aggressive compression brings the full submission (model + code) to 15.1MB, under the 16MB cap.

## Metrics

|Metric|Value|
|-|-|
|**Post-quant sliding window val\_bpb**|**1.1556**|
|Post-quant sliding window val\_loss|1.9511|
|Post-quant standard val\_bpb|1.1891|
|Post-quant standard val\_loss|2.0077|
|Quantization gap (standard eval)|\~0.0001 BPB|
|Model parameters|22,368,840|
|Artifact size (int6+zstd-22)|15,878,809 bytes (15.1 MB)|
|Train steps completed|12,047|
|Train time|600s (10.0 min)|
|Sliding window eval time|75s|
|Peak GPU memory|11,340 MiB|

## Configuration

```
VOCAB\_SIZE=1024
NUM\_LAYERS=9
MODEL\_DIM=512
NUM\_HEADS=8
NUM\_KV\_HEADS=4
MLP\_MULT=3
TIE\_EMBEDDINGS=1
USE\_SMEARGATE=1
TRAIN\_SEQ\_LEN=1024
TRAIN\_BATCH\_TOKENS=524288
LOGIT\_SOFTCAP=30.0
ROPE\_BASE=10000.0
QK\_GAIN\_INIT=1.5
BIGRAM\_HASH\_BUCKETS=4096
BIGRAM\_HASH\_DIM=128
TIED\_EMBED\_LR=0.030
MATRIX\_LR=0.020
SCALAR\_LR=0.020
MUON\_MOMENTUM=0.99
MUON\_MOMENTUM\_WARMUP\_START=0.92
MUON\_MOMENTUM\_WARMUP\_STEPS=1500
MUON\_WEIGHT\_DECAY=0.01
MUON\_BACKEND\_STEPS=5
WARMDOWN\_ITERS=3000
WARMUP\_STEPS=20
EVAL\_STRIDE=64
MAX\_WALLCLOCK\_SECONDS=600
SEED=1337
```

## Command

```bash
RUN\_ID=smeargate\_orthoinit\_muonwd \\
DATA\_PATH=./data/datasets/fineweb10B\_sp1024 \\
TOKENIZER\_PATH=./data/tokenizers/fineweb\_1024\_bpe.model \\
torchrun --standalone --nproc\_per\_node=8 train\_gpt.py
```

## Hardware

8× NVIDIA H100 80GB HBM3 SXM (RunPod).

## 

