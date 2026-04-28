# Report: PTQ int6-attn + int5-mlp, L=20, mlp=5

## Approach

- **Architecture**: GPT-like model with 20 layers (`num_layers=20`), `model_dim=256`, MLP multiplier 5, GQA (8 heads, 4 KV-heads), tied embeddings, vocab_size=1024
- **Quantization (PTQ)**: post-training quantization — int6 for attention weights, int5 for MLP weights. QAT was not used (`qat_bits=0`)
- **Optimizer**: Muon + Adam (`muon_adam`) with trapezoid LR schedule, warmup 20 steps, warmdown 1200 iterations
- **Learning rates**: embed_lr=0.6, head_lr=0.008, matrix_lr=0.04, scalar_lr=0.04, tied_embed_lr=0.05
- **Data**: FineWeb 10B, SentencePiece tokenizer (1024 tokens), batch 524288 tokens, seq_len=1024
- **Training**: 8×GPU (distributed), wallclock limit 600 seconds (10 minutes), ~4500 iterations completed
- **Compilation**: `torch.compile` enabled for both model and Muon optimizer
- **FlashAttention**: not used (flash-attn package unavailable, fallback to torch SDPA)
- **Serialization**: int8 quantization + zlib compression for the final checkpoint

## Results

### Validation (final, after int8 zlib roundtrip)

| Metric | Value |
|--------|-------|
| **val_loss (full precision)** | 2.1657 |
| **val_bpb (full precision)** | 1.2827 |
| **val_loss (quantized model)** | 2.2432 |
| **val_bpb (quantized model)** | 1.3286 |

### Model Size

| Artifact | Size |
|----------|------|
| FP32 checkpoint (`final_model.pt`) | **66 MB** (68,848,491 bytes) |
| Int8 + zlib (`final_model.int8.ptz`) | **15 MB** (15,416,793 bytes) |
| Int8 payload (before compression) | **21 MB** (21,389,952 bytes) |
| Code (`train_gpt_lib/` + `train_gpt.py` + `main.py`) | **~92 KB** (94,353 bytes) |
| **Total (compressed model + code)** | **~15.1 MB** |
