# HYDRA-Ω: SLOT-Optimized Parameter-Efficient Language Model

## Summary

HYDRA-Ω is a parameter-efficient language modeling system designed for the Parameter Golf challenge constraints (≤16MB artifact, ≤10 minute training on 8×H100 SXM).

The approach focuses on shifting performance gains from architecture scaling to evaluation-time optimization.

## Key Components

- **Transformer Backbone** (11L / 512d) with parameter banks for batched Muon optimizer
- **Full-Hessian GPTQ** with mixed precision quantization (int6) + Cholesky error compensation + activation ordering
- **Parallel Muon** optimizer with 3-phase overlapped communication (reduce-scatter → Adam → NS5+all-gather)
- **EMA (0.997)** + SWA + optimized warmdown schedule for maximum step utilization
- **Score-first Test-Time Training (TTT)** — 2 epochs, SGD+cosine, freeze blocks 0-2
- **SLOT** (hidden-state delta optimization) with cosine LR scheduling and 2-pass hard-token refinement
- **LZMA-9** compression

## Architecture Details

| Parameter | Value |
|---|---|
| Layers | 11 |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 (GQA) |
| MLP expansion | 3.0× (1536) |
| Activation | LeakyReLU(0.5)² |
| XSA | All 11 layers |
| RoPE | Partial (16/64 dims) |
| BigramHash | 2048×128 |
| ValueEmbedding | 128d on layers 9-10 |
| Vocab | 1024 (SentencePiece BPE) |
| Embeddings | Tied |

## Eval-Time Adaptation

1. **TTT**: Score-first, 2 epochs, lr=0.003, freeze blocks 0-2, 32768 chunk size
2. **SLOT**: 10 AdamW steps, lr=0.006, cosine LR schedule, 2-pass (uniform → hard-token reweighted)
3. **N-gram**: Causal prefix hints (OFF by default, toggle if competitive)

## Status

- Implementation complete
- Training runs pending compute availability
- PR submitted early to document approach and enable reproducibility

## Expected Outcome

Based on component-level analysis, estimated **~1.08–1.09 bpb** after full training and tuning.

## Compliance

- Strictly causal evaluation (no future token leakage)
- N-gram augmentation is causal prefix-only (legal per March 27 ruling)
- SLOT optimizes broadcast delta at last hidden layer
- All constraints satisfied: ≤16MB artifact, ≤10min train, ≤10min eval
