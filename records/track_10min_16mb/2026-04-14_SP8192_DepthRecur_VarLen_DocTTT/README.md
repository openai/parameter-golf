# SP8192 + Depth Recurrence + VarLen Attention + Doc-LoRA TTT

## Architecture

- **Tokenizer**: SP8192 (SentencePiece BPE 8192 vocab) — 8x larger vocab for better token efficiency
- **Model**: 11 layers, 512d, 8 heads / 4 KV heads (GQA), MLP 3x (LeakyReLU(0.5)^2)
- **Depth Recurrence**: Layers 3-5 repeat 2 passes each with learned gated blending
- **Parallel Residuals**: GPT-J style for layers 7+ (attention and MLP in parallel with learned gate)
- **QK-Gain 5.25**: Amplified attention sharpness via per-head gain scaling
- **BigramHash 3072×112**: Context-aware token hashing with XOR hash
- **XSA on all 11 layers**: Orthogonal value-subspace attention removal
- **SmearGate**: Learned temporal smoothing between consecutive positions
- **Value Embedding (VE128)**: Layers 9-10 get value-stream shortcuts
- **Partial RoPE (16/64)**: Only 16 of 64 head dims get positional encoding
- **U-Net Skip Connections**: Encoder-decoder skip pattern between layers
- **Tied Embeddings**: Input/output embedding sharing

## Training

- **Muon Optimizer**: Parallel reduce-scatter + Newton-Schulz orthogonalization, momentum=0.97
- **EMA**: Exponential moving average with decay=0.997
- **Tight SWA**: Stochastic weight averaging every 50 steps when LR < 20%
- **Late QAT**: Quantization-aware training activated when LR scale < 0.15
- **Warmdown 0.75**: Wall-clock-aware cosine warmdown over 75% of training time
- **Weight decay**: Muon WD=0.095, Adam WD=0.04

## Quantization

- **Full Hessian GPTQ int6**: AR self-generated calibration (64 seqs × 2048 tokens, temp=0.8)
  - Cholesky decomposition with column reordering
  - 5-percentile clip search for optimal scale
  - Block-size 128 error compensation
- **Selective ±1 pruning**: Binary search to fit 16MB target
- **LZMA preset=9** compression

## Evaluation

- **Score-First AdamW Doc-LoRA TTT**: Novel document-aware test-time training
  - Fresh LoRA (rank=8) per document, all 11 layers
  - Each chunk scored FIRST under no_grad, THEN LoRA adapted
  - AdamW optimizer (β1=0.9, β2=0.999) instead of SGD for per-parameter adaptive learning
  - Chunk size = 64 tokens with context window
  - Fully legal under Issue #1017 (score-before-update, single L→R pass)
- **Sliding window eval**: Stride=64 for reliable BPB measurement

## Key Innovations

1. **AdamW TTT**: First use of AdamW (not SGD) for test-time LoRA adaptation — adaptive per-parameter learning rates improve adaptation quality
2. **Depth Recurrence + Parallel Residuals**: Combining repeated layer processing with parallel residual streams for parameter-efficient compute
3. **SP8192 vocab**: 8x larger vocabulary captures subword patterns more efficiently than SP1024
4. **Gated recurrence blending**: Learned gates prevent catastrophic overwriting during layer repetition

## Expected Performance

Target: ~1.058-1.065 BPB (vs current best 1.11564)
