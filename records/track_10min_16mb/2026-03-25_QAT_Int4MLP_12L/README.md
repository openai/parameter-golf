# 12L QAT Int4-MLP + Int6-Attn on PR #549 Stack

**val_bpb: TBD** (post-quant int4/int6+lzma, sliding window stride=32, legal TTT)

## Summary

Built on the #1 SOTA (1.1194, PR #549 by @abaybektursun). Adds full QAT with mixed-precision fake-quantization (int4 for MLP, int6 for attention) and uses the byte savings to fund a 12th transformer layer.

## Key Changes from SOTA (1.1194)

### 1. Quantization-Aware Training (QAT) with Mixed Precision
Full QAT via Straight-Through Estimator applied to banked weights during every forward pass:
- **MLP weights: int4** (clip_range=7) — MLPs are less precision-sensitive
- **Attention weights: int6** (clip_range=31) — attention is more sensitive
- Applied directly in MLP.forward and CausalSelfAttention.forward (not via CastedLinear)

### 2. Int4 MLP Post-Training Quantization
Post-training quantization uses GPTQ-lite clip search with int4 for MLP (clip=7) and int6 for attention (clip=31). Int4 compresses ~3x with LZMA vs ~1.6x for int6, saving ~3MB.

### 3. 12th Transformer Layer
The byte savings from int4 MLP fund an extra layer (12 vs 11). Architecture: 6 encoder + 6 decoder with U-Net skip connections.

### 4. Eval Stride 32 (was 64)
Halved sliding window eval stride for more context overlap per scored token.

## Inherited Techniques (from PR #549 stack)
- LeakyReLU(0.5)^2 activation
- Legal Score-First TTT (3 epochs SGD per 32K chunk)
- Parallel Muon optimizer with Parameter Banking
- XSA on last 4 layers
- Partial RoPE (16/64 dims)
- LN Scale 1/sqrt(layer+1)
- EMA (decay=0.997)
- SmearGate + BigramHash(2048)
- Value Embedding on layers 9,10
- LZMA compression

## Architecture
- 12 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (1536), LeakyReLU(0.5)^2
- ~29M params, estimated artifact ~15MB

## Run Command

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Improvement
- 12th layer: est. -0.002 to -0.003 bpb (based on 10L->11L gains)
- QAT reducing quant penalty: est. -0.001 to -0.002 bpb
- Stride 32 eval: est. -0.001 bpb
- Net target: ~1.114-1.117 bpb
