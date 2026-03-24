## Record: 11L EMA + GPTQ-lite + LoRA TTT

**Architecture**: Full PR #401 (11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15)
**Eval innovation**: Per-document LoRA test-time training at evaluation

### The Insight

| Component | val_bpb | Source |
|-----------|---------|--------|
| Current SOTA (PR #401) | 1.1228 | Best architecture + training, ZERO test-time adaptation |
| LoRA TTT on naive baseline | 1.1928 | Naive model + per-doc LoRA TTT (0.031 improvement from 1.2244) |
| **Combined (this)** | **TBD** | Best architecture + LoRA TTT. Nobody has combined them. |

The LoRA TTT record showed that per-document test-time training gives ~0.03 bpb improvement on the naive baseline. That improvement came from three sources:
- Document isolation: -0.011 bpb
- Strided evaluation: -0.023 bpb
- LoRA adaptation: -0.003 bpb

The current SOTA already uses sliding window eval (stride=64), so the strided eval gains are already captured. The remaining free lunch is **document isolation + LoRA adaptation**.

### What Changed vs PR #401

**Training**: IDENTICAL. No changes to training, architecture, quantization, or model size.

**Evaluation only**:
1. Added `forward_with_lora` method to GPT class (accepts per-batch LoRA adapters for Q/V projections)
2. Modified `CausalSelfAttention.forward` and `Block.forward` to accept optional `q_delta`/`v_delta` LoRA modules
3. Added `BatchedTTTLoRA` class: rank-8 LoRA adapters for Q, V, and (optionally) LM head
4. Added `eval_val_ttt_lora`: per-document chunked evaluation with LoRA adaptation

### TTT Eval Method

For each document in the validation set:
1. Find document boundaries using BOS tokens
2. Split into chunks (chunk_size=256) within sliding context windows (eval_seq_len=2048)
3. For each chunk: score tokens, then train LoRA on that chunk's loss (no leakage)
4. Reset LoRA parameters between documents
5. Documents batched (batch_size=32) and sorted by length for GPU efficiency

### TTT Hyperparameters

| Param | Value |
|-------|-------|
| LoRA rank | 8 |
| LoRA targets | Q, V projections (all 11 layers) + LM head (if untied) |
| Learning rate | 0.01 |
| Optimizer | Adam (betas=0.9, 0.95) |
| Chunk size | 256 |
| Eval seq len | 2048 |
| Batch size | 32 |

### Run Command

```bash
USE_TTT=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To disable TTT and get standard sliding window eval only:
```bash
USE_TTT=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
