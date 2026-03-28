# Non-Record Submission: 1.3712 BPB — Value Embeddings + 8L 512d (1×RTX 4090)

**Score:** 1.3712 BPB
**Parameters:** 50.3M
**Hardware:** 1× NVIDIA RTX 4090 (24GB)
**Training time:** ~301s (5 min)
**Total time (incl. eval):** ~388s

## Architecture

- 8-layer transformer, 512 embedding dim, 4 attention heads, 4 KV heads
- Value Embeddings (16.7M params) with gated fusion on alternating layers
- Tied input/output embeddings
- Window pattern: SSSL (sliding + sliding + sliding + long)
- Vocab: 8192 BPE
- Sequence length: 2048
- MLP ratio: 4x
- AdamW + Muon optimizer with cosine LR schedule
- Gradient accumulation: 16 microbatches

## Key Technique: Value Embeddings

The model uses a large value embedding table (16.7M params) that provides additional context to the attention mechanism via gated fusion on alternating layers. This is the primary source of the model's strong performance relative to its transformer parameter count.

## Config

```python
{'sequence_len': 2048, 'vocab_size': 8192, 'n_layer': 8, 'n_head': 4, 'n_kv_head': 4, 'n_embd': 512, 'window_pattern': 'SSSL'}
```

## Parameter Breakdown

- wte (token embeddings): 4,194,304
- value_embeds: 16,777,216
- lm_head (tied): 4,194,304
- transformer_matrices: 25,166,336
- scalars: 16
- **Total: 50,332,176**

## Training Details

- 107 steps, batch size via gradient accumulation of 16
- Peak VRAM: 11.8 GB
- MFU: 4.08%
- Throughput: ~168K tok/sec
- LR scaled by 1/sqrt(512/768) = 1.2247
- Final training loss: 3.922

## Eval

- int8 zlib roundtrip exact evaluation
- val_bpb: **1.3712**

## Notes

This is a non-record submission trained on a single consumer GPU (RTX 4090) rather than 8xH100. The value embedding technique is interesting and could potentially be combined with other winning techniques (SmearGate, XSA, int6 QAT, sliding window eval) to push scores lower on the full compute budget.

## Author

ivanontech
