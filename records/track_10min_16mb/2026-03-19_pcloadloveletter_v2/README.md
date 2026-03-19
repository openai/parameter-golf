# pcloadloveletter v2

Submission for the OpenAI Parameter Golf challenge, 10min/16MB track.

## Base

Built on `2026-03-19_SlidingWindowEval/train_gpt.py` which provides loop support, LoRA scaffolding, QAT scaffolding, and sliding window evaluation.

## Techniques

### Int6 Quantization

Replaces int8 per-row quantization with int6: `scale = max(abs(row)) / 31`, values clamped to `[-32, 31]`. Stored in int8 containers for PyTorch compatibility. This gives ~25% size reduction over int8 at a small accuracy cost, which is partially mitigated by keeping critical tensors in fp16.

### Late-K Passthrough

The last 2 transformer layers' K-projection weights (`blocks.7.attn.c_k.weight` and `blocks.8.attn.c_k.weight`) are kept in fp16 instead of being quantized. These late-layer attention keys are disproportionately important for output quality.

### tok_emb.weight in fp16

The token embedding table is kept in fp16 rather than quantized, preserving embedding quality.

### zstd Compression (level 22)

Replaces zlib level 9 with zstandard level 22 for better compression ratios on the quantized model blob. Falls back to zlib if `zstandard` is not installed.

### MLP 3x Width

MLP hidden dimension increased from 2x (1024) to 3x (1536) model_dim. More expressive feedforward layers within the same parameter budget tradeoff.

### SmearGate

A cheap (~512 parameter) learned temporal smoothing module applied after embedding normalization and before the first transformer block. Computes `x = (1 - gate) * x + gate * x_prev` where `x_prev` is x shifted right by 1 position and `gate = sigmoid(learned_param)`. Initialized at 0 (sigmoid(0) = 0.5). Helps the model capture local token dependencies cheaply.

### Sliding Window Eval

Evaluation uses a sliding window with stride=64 and batch_seqs=1024. Each token is scored with maximum available context, giving a more accurate BPB estimate than fixed-chunk evaluation.

### Tuned Hyperparameters

- `TRAIN_SEQ_LEN=2048` (up from 1024)
- `MATRIX_LR=0.02` (down from 0.04)
- `SCALAR_LR=0.02` (down from 0.04)
- `TIED_EMBED_LR=0.04` (down from 0.05)
- `MUON_MOMENTUM=0.99` (up from 0.95)
- `MUON_MOMENTUM_WARMUP_START=0.92` (up from 0.85)
- `MUON_MOMENTUM_WARMUP_STEPS=1500` (up from 500)
- `WARMDOWN_ITERS=3000` (up from 1200)
- `QAT=0` (disabled; int6 post-training quantization is sufficient)
- `NUM_LOOPS=1`, `LORA_RANK=0` (single pass, no LoRA)

## Running

```bash
torchrun --nproc_per_node=8 train_gpt.py
```

Requires `zstandard` pip package for zstd compression (falls back to zlib otherwise).
