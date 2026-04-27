# SP8192 Depth Recurrence + Parallel Residuals + TTT

## Configuration
- **Tokenizer:** SP8192 (8192 vocab SentencePiece BPE)
- **Architecture:** 11 physical layers, 512 dim, 8 heads / 4 KV heads
- **Depth recurrence:** Layers 3-5 shared, looped 3x = 17 virtual layers
- **Parallel residuals:** GPT-J style for layers 7+
- **Activation:** LeakyReLU(0.5)^2
- **Partial RoPE:** 16/64 head dims with rotary, rest position-free
- **QK gain:** 5.25 (learnable per-head)
- **Skip connections:** U-Net encoder-decoder with sigmoid-gated skip weights

## Training
- **Optimizer:** Muon (matrices) + Adam (embeddings/scalars)
- **Batch:** 524,288 tokens/step on 8xH100
- **Warmdown:** 72% linear warmdown
- **EMA:** Decay 0.9965, starts at 50% of training
- **Quantization:** int8 per-row + zlib level 9

## Evaluation
- **Test-Time Training:** Score-first chunk-based SGD (lr=0.005, 3 epochs, momentum=0.9)

## Results (8xH100, 10 min)
- val_bpb: **1.1921** (without TTT)
- val_bpb: **1.1888** (with TTT)
- Artifact: 15.79 MB (PASS)
