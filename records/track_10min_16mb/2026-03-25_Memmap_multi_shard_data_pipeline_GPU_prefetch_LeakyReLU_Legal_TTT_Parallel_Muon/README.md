# Memmap multi-shard data pipeline + GPU prefetch + LeakyReLU² + Legal TTT + Parallel Muon

**val_bpb: 1.1147** (3-seed mean, std 0.0005) | **~15.23 MB** (mean int6+lzma + code) | 8×H100 80GB HBM3

## Results (8×H100 80GB HBM3, PyTorch 2.10.0+cu126)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | TTT time | Artifact |
|------|----------|-------|-------------|------------------|----------|----------|----------|
| 1337 | 83.1ms | 7,223 | 1.1171 | **1.1140** | -0.0031 | 385s | 15,977,541 |
| 1111 | 83.0ms | 7,227 | 1.1178 | **1.1149** | -0.0029 | 383s | 15,964,369 |
| 13 | 83.0ms | 7,226 | 1.1179 | **1.1151** | -0.0028 | 386s | 15,957,041 |
| **Mean** | **83.1ms** | **7,225** | **1.1176** | **1.1147 (std 0.0005)** | **-0.0029** | **~384s** | |

*Pre-TTT bpb* is `final_int6_sliding_window_exact` (GPTQ-lite int6 dequantized weights, sliding window, stride 64). *Post-TTT bpb* is `legal_ttt_exact` after score-first chunk adaptation. *Artifact* is `Total submission size int6+lzma` (compressed weights + UTF-8 `train_gpt.py` bytes).

## Key changes

### 1. Training data pipeline

The old loader walked shards in fixed order and took contiguous spans per rank. The new `DistributedTokenLoader`:

- Memory-maps each `.bin` shard (`numpy.memmap`) with a small header-token cache to avoid repeated header reads.
- Samples **global** training windows across shards: per batch, it draws multiple shards with probability weighted by remaining usable blocks (exponent tightens as training progresses), uses a **coprime stride** over valid 2048-token blocks inside each shard, and interleaves windows for batch diversity.
- Merges nearby reads on the same shard into one slab copy to reduce mmap churn.
- Overlaps I/O and compute: a **daemon thread** builds CPU `(x, y)` batches into a queue while the main thread trains; **CUDA streams and events** prefetch the next batch to GPU while the current step runs.

Together this replaces the old strictly sequential per-rank stream with a stratified, multi-shard mixture and async H2D overlap. Model forward, optimizer (Parallel Muon + AdamW), EMA/SWA, late QAT, and export paths are otherwise in the same family as the previous script.

### 2. Defaults and TTT

- `TTT_ENABLED` defaults to **1** in `train_gpt.py` (it was **0** in `train_gpt_old.py`), so legal score-first TTT runs at the end unless disabled.
- `TTT_FREEZE_BLOCKS` defaults to **2** for the logged runs (`freeze_blocks=2` in `ttt_sliding:start` in the training logs). The prior README’s command used `TTT_FREEZE_BLOCKS=0` (all blocks); this submission’s numbers are with **two frozen bottom blocks** during TTT unless you override the env var.

### 3. LeakyReLU² (unchanged from prior submission)

Same MLP nonlinearity as before:

```python
x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
return F.linear(x.square(), down_w.to(x.dtype))
```

LeakyReLU(0.5) keeps gradients on negative pre-activations; squaring preserves the non-negative “relu²-style” bias.

## Legal TTT protocol

Unchanged in spirit from the previous write-up (backward-looking, score-first, PR #549-style):

1. Val tokens split into non-overlapping 32K-token chunks (1,893 chunks at the logged val size).
2. **Per chunk:** **SCORE** all assigned sliding windows under `torch.inference_mode()`; then **TRAIN** on that chunk with SGD(momentum), cosine LR across chunks, 3 epochs, grad clip 1.0.
3. Last chunk is scored but not trained on; chunk *N* is scored only under weights adapted on chunks `0..N-1`.

### TTT hyperparameters (as logged)

| Parameter | Value |
|-----------|-------|
| Chunk size | 32,768 tokens |
| Optimizer | SGD + momentum(0.9) |
| Learning rate | 0.002 (cosine decay across chunks) |
| Epochs per chunk | 3 |
| Frozen blocks | **2** (first two transformer blocks frozen) |
| Gradient clip | 1.0 |
| Sliding stride | 64 |

### Timing budget (representative, seed 1337)

| Phase | Time |
|-------|------|
| Training | 600s (wall-clock cap) |
| Standard eval (int6 roundtrip + sliding window stride 64) | ~75s |
| Legal TTT (score-first sliding + adaptation) | ~384s |
| **Total eval** | **~460s (< 10 min)** |

## Training architecture

Same PR #549-style stack as before: 11 layers (512d, 8H, 4KV), encoder/decoder skip wiring, BigramHash, XSA on the last four layers, partial RoPE (16/64), per-layer LN scale `1/√(layer+1)`, value embeddings on layers 9–10, EMA(0.997) + tight SWA (every 50 steps once warmdown LR scale drops below 0.2), GPTQ-lite int6 + lzma for submission bytes.

## Run command

Matches the logged configuration (8 GPUs, 600s cap, TTT on, default freeze=2).

```bash
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`Hyperparameters` defaults used in the logs.


## Credits

- **LeakyReLU² + Legal Score-First TTT + Parallel Muon**: Accepted PR #549.