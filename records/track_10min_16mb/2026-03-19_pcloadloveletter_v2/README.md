# pcloadloveletter v4

Submission for the OpenAI Parameter Golf challenge, 10min/16MB track.

## Base

Built on `2026-03-19_SlidingWindowEval/train_gpt.py` which provides loop support, LoRA scaffolding, QAT scaffolding, and sliding window evaluation.

## Changes from v3

### 11 Layers (up from 9)

Increased transformer depth from 9 to 11 layers. The int6+zstd compression budget accommodates the extra parameters. Skip connection weights automatically adjust via `effective_depth // 2` (5 encoder, 6 decoder, 5 skip weights). Late-K passthrough updated to blocks.9 and blocks.10.

### BigramHash Embedding

New `BigramHashEmbedding` module adds learned bigram features to the token embeddings. Uses a hash function `XOR(36313 * t[i], 27191 * t[i-1]) % (vocab_size - 1)` to map consecutive token pairs to a 2048-entry embedding table (128 dims), projected to model_dim with a learnable scale (init 0.05). Zero-initialized so training starts from the unigram baseline. Embed weights go to Adam (token LR), proj weight to Muon, scale to scalar Adam.

### Weight Decay on Muon (0.04)

Added decoupled weight decay to the NorMuon optimizer. Applied as `p.mul_(1 - lr * weight_decay)` before the Muon update step. Helps regularize the large matrix parameters.

### Orthogonal Initialization

All CastedLinear weights with `min(shape) >= 64` are initialized with `nn.init.orthogonal_(gain=1.0)` instead of PyTorch's default Kaiming uniform. Zero-init modules (output projections) are preserved. Orthogonal init provides better gradient flow at initialization.

### SWA Every 50 Steps (down from 200)

Stochastic Weight Averaging now collects checkpoints every 50 steps instead of 200, providing more snapshots during the warmdown phase for a better averaged model.

### RoPE Base 50K (up from 10K)

Rotary position embedding base frequency increased from 10,000 to 50,000. With TRAIN_SEQ_LEN=2048, the higher base provides smoother position encoding across the sequence.

### Eval Stride 64 (down from 256)

Sliding window evaluation stride reduced to 64 for more accurate BPB scoring. Each token gets scored with near-maximum context. EVAL_BATCH_SEQS=64 keeps memory usage reasonable on 8xH100.

## Techniques (inherited from v3)

- Int6 quantization ([-31, 31] in int8 containers) with outlier clipping
- zstd level 22 compression
- Late-K passthrough (last 2 layers' K proj in fp16)
- tok_emb.weight in fp16
- SmearGate (learned temporal smoothing before first block)
- MLP hidden=1500
- Tied embeddings (init std=0.005)
- Logit softcap=30
- NorMuon optimizer (per-row second-moment normalization)
- TRAIN_SEQ_LEN=2048, TRAIN_BATCH_TOKENS=786432
- WARMDOWN_ITERS=3000
- Grad clip norm=0.3

## Running

```bash
torchrun --nproc_per_node=8 train_gpt.py
```

Requires `zstandard` pip package for zstd compression (falls back to zlib otherwise).
