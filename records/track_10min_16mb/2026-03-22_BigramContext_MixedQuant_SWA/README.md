# Bigram-Aware Context Modeling with Mixed-Precision Quantization

**val_bpb: 1.1431** | **15.97 MB** | 8xH100 SXM, 600s train + 175s eval

---

## Overview

This submission explores how injecting explicit bigram-level context into the embedding pipeline — combined with a quantization strategy that allocates precision based on layer sensitivity — can push a 10-layer, 25.5M-parameter transformer to strong compression performance under the 16MB artifact limit.

The core insight is that standard token embeddings discard local context: the representation for token `t` at position `i` is identical regardless of what precedes it. Recovering this bigram signal through self-attention is expensive in both parameters and depth. We provide it directly through two lightweight mechanisms, freeing attention capacity for longer-range dependencies.

---

## Architecture

### Model Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Layers | 10 | Maximum depth that fits under 16MB with mixed int5/int6 quantization |
| Model dim | 512 | Standard width; wider models hit the artifact cap too quickly |
| Attention heads | 8 (4 KV) | Grouped-query attention halves KV memory with negligible quality loss |
| MLP expansion | 3x (hidden=1536) | MLP weights compress well at int5; 3x beats 2x by using the freed budget |
| Seq length | 2048 | Longer context improves per-token prediction; 2048 is the sweet spot between context and step throughput |
| Vocab size | 1024 | SP-1024 BPE tokenizer (fixed by challenge dataset) |
| Total params | 25,517,137 | |

### Bigram Context: Why and How

A transformer must discover that "th" is commonly followed by "e" through learned attention patterns. At this model scale (10 layers, 512 dim), dedicating attention capacity to such local patterns is wasteful. We inject bigram context through two complementary mechanisms:

**BigramHash Embedding.** We hash consecutive token pairs via `(36313 * t_cur) XOR (27191 * t_prev) mod 10239` into a 10,240-bucket learned embedding table (dim=128), then project to model dimension. The XOR hash was chosen for uniform bucket distribution and low collision rate. The table is zero-initialized with a learned scale starting at 0.05, allowing the model to gradually incorporate bigram information as training progresses. This adds ~1.3M parameters but provides the model with explicit token-pair identity that would otherwise require multiple attention layers to recover.

**SmearGate.** A learned per-dimension sigmoid gate blends each token's embedding with the previous token's embedding: `out = (1-g)*x + g*x_prev`. Initialized at `sigmoid(0) = 0.5` for equal blending, the gate learns per-dimension which features benefit from local smoothing versus sharp token boundaries. This captures soft continuous bigram context, complementing the discrete hash lookup.

Together, these mechanisms give every subsequent attention layer access to local context at near-zero computational cost (~0.3ms overhead per step).

### Depth and Width Tradeoffs

The 16MB artifact constraint forces a fundamental tradeoff between depth and width. We chose 10 layers with 3x MLP expansion based on a key observation about quantization asymmetry:

- **MLP weights** have smooth, concentrated distributions. At int5 (32 levels), they achieve 1.88x compression under zstd — aggressive quantization with minimal quality loss.
- **Attention weights** have broader distributions with more outliers. They require int6 (64 levels) for acceptable quality, achieving 1.51x compression.

By quantizing MLP weights to int5 instead of uniform int6, we save ~1.86MB — enough to fund an entire 10th transformer layer. The capacity gain from the additional layer (~0.01 bpb) far exceeds the quality loss from coarser MLP quantization (~0.002 bpb).

### U-Net Skip Connections

The 10 layers are split into a 5-layer encoder and 5-layer decoder with learned per-dimension skip weights. This architecture allows the decoder to directly access encoder representations at matching depth, improving gradient flow and enabling the model to combine low-level token features with high-level semantic representations.

### Residual Mixing

Each block blends the running hidden state with the original post-embedding representation via a learned 2D mixing parameter. This gives each layer the option to partially "reset" toward the original signal, counteracting representation drift in deeper layers without the cost of additional normalization layers.

---

## Training

### Muon Optimizer with Weight Decay

Matrix-shaped parameters are optimized with Muon, which orthogonalizes gradients via Newton-Schulz iteration before applying updates. The orthogonalization produces weight matrices with tighter singular value distributions, which directly benefits post-training quantization by reducing outlier magnitudes.

Decoupled weight decay at 0.04 serves dual purpose: regularization during training, and keeping weight magnitudes compact for quantization. Lower values (0.01-0.02) leave too many outliers that expand quantization scales; higher values (0.08+) hurt convergence.

### Momentum Schedule

Muon momentum warms from 0.92 to 0.99 over 1,500 steps. Lower initial momentum allows larger exploratory updates early in training when the loss landscape is rough. High momentum late in training smooths the optimization trajectory toward a flat minimum — critical for both SWA convergence and quantization robustness.

### Learning Rate Warmdown

Linear warmdown over the final ~3,000 steps (wall-clock based, approximately the last 45% of training). This gradually reduces the learning rate to near-zero, allowing SWA to collect checkpoints from a progressively stabilizing region of weight space.

### Gradient Clipping

Global gradient norm clipping at 0.3 prevents occasional gradient spikes from destabilizing training, particularly important with the Muon optimizer where Newton-Schulz orthogonalization can amplify certain gradient directions.

---

## Post-Training Pipeline

### Stochastic Weight Averaging (SWA)

SWA collects a checkpoint every 50 steps during the last 40% of the warmdown schedule, averaging all collected checkpoints to produce the final model weights. In this run, 24 checkpoints were averaged.

The averaged weights occupy a wider, flatter region of the loss landscape with two concrete benefits:
1. **Fewer weight outliers** — the averaging process cancels noise, producing smoother weight distributions that quantize more accurately.
2. **Better compression** — SWA-smoothed weights compress approximately 15% smaller under zstd compared to non-averaged final weights.

### Magnitude Pruning

Before quantization, the smallest 3% of weights by absolute value in all large 2D matrices are set to zero. These near-zero weights contribute negligibly to model output but create unnecessary entropy in the quantized representation. Zeroing them produces runs of identical bytes that compress efficiently.

### Mixed-Precision Quantization

| Weight Category | Precision | Clip Range | Rationale |
|----------------|-----------|------------|-----------|
| MLP (fc, proj) | Int5 | [-16, 15] | Smooth distributions; 32 levels sufficient |
| Attention (Q, K, V, proj) | Int6 | [-32, 31] | Broader distributions; precision-sensitive |
| Token embeddings | FP16 | Full | Dual role (input + output) means errors compound bidirectionally |
| Layer 8 key projection | FP16 | Full | Empirically sensitive; penultimate layer's keys affect all subsequent attention |
| Control tensors | FP32 | Full | Scales, gates, mixing weights — few parameters, high sensitivity |

Per-row scales (FP16) allow each output channel to use its own dynamic range, capturing per-neuron magnitude variation without global scale bottlenecks.

### Compression

zstd at level 22 provides approximately 5% better compression than zlib-9, at the cost of ~2 additional seconds of serialization time. Combined with int5/int6 quantization and magnitude pruning, the final artifact is 15.97MB.

---

## Evaluation

### Sliding Window (Stride=64)

Standard evaluation partitions the validation set into non-overlapping chunks, giving each token an average of seq_len/2 tokens of context. Sliding window evaluation advances by only 64 tokens per window, scoring only the rightmost 64 tokens (first window scores all positions). This gives every scored token at least 1,984 tokens of context.

This is purely an evaluation-time technique with zero artifact cost. The only tradeoff is computation: ~175 seconds versus ~16 seconds for standard evaluation, well within the 10-minute eval budget.

---

## Results

| Metric | Value |
|--------|-------|
| **val_bpb (post-quant roundtrip)** | **1.14308** |
| val_loss (post-quant roundtrip) | 1.93003 |
| Pre-quant val_bpb | 1.1533 |
| Quantization gap | 0.010 bpb |
| Training steps | 6,658 |
| Step time | ~90 ms |
| Training wall-clock | 600s |
| Eval wall-clock | 175s |
| SWA checkpoints | 24 |
| Artifact size | 15,971,094 bytes |

### Training Trajectory

| Step | val_bpb | train_loss |
|------|---------|------------|
| 0 | 4.1057 | 6.9334 |
| 500 | 1.3933 | 2.3934 |
| 1,000 | 1.3182 | 2.2737 |
| 2,000 | 1.2633 | 2.0683 |
| 3,000 | 1.2391 | 2.1585 |
| 4,000 | 1.2273 | 1.9805 |
| 5,000 | 1.2000 | 2.1069 |
| 6,000 | 1.1723 | 1.9423 |
| 6,658 | 1.1533 | — |

---

## Reproducibility

```bash
RUN_ID=run_seed42 SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=3.0 TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 WEIGHT_DECAY=0.04 GRAD_CLIP_NORM=0.3 \
BIGRAM_VOCAB_SIZE=10240 BIGRAM_DIM=128 \
SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 \
EMA_ENABLED=0 XSA_LAST_N=0 ROPE_DIMS=0 LN_SCALE=0 \
EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 PRUNE_FRAC=0.03 \
MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
