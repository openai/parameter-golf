# Soft MoE: Exploring Mixture of Experts Under the 16MB Constraint

## Summary
Non-record submission exploring whether Mixture of Experts (MoE)
architectures can improve parameter golf performance. Key finding:
standard sparse MoE fails under 16MB constraints, but a dense
"Soft MoE" variant fixes all identified problems.

**Best result:** val_bpb = 1.1826 (11L Soft MoE, 8xH100, 600s) —
artifact was 17.3MB (over limit). 10L version expected to fit under 16MB.
Work in progress.

## Approach

### What Failed: Sparse MoE
- **Router collapse:** 98% of tokens routed to one expert, even with
  10x aux loss coefficient (0.1 vs default 0.01)
- **torch.compile incompatibility:** Variable-size tensors from sparse
  dispatch caused constant recompilation. Step time: 2309ms (vs 794ms baseline)
- **Parameter overhead:** Full-size experts doubled MLP params without
  proportional quality gain

### What Worked: Soft MoE
Dense gating where ALL experts run on ALL tokens with learned soft weights.
- **No collapse possible** — both experts always receive gradients
- **Compile-friendly** — no variable-size tensors, enables torch.compile
- **Step time: 636ms** — faster than baseline due to smaller individual experts
- **1.1826 bpb** on 11L config (vs 1.2244 baseline)

### Architecture
- 10-11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- Soft MoE on last 2 layers only (MOE_START_LAYER=8 or 9)
- 2 experts per MoE layer, each with mlp_mult/2 hidden dim
- SmearGate + BigramHash(10240, dim=128)
- EMA (decay=0.998) replacing SWA
- Int5 MLP / Int6 attention quantization + zstd-22
- Sliding window eval stride=64

### Experiment Log

| Test | Config | Steps | val_bpb | Step/ms | Artifact | Expert Balance |
|------|--------|-------|---------|---------|----------|----------------|
| 1 | 11L no MoE | 138 | 3.26 | 794 | 17.5MB | n/a |
| 2 | 9L sparse MoE 2exp | 52 | 3.86 | 2309 | too big | n/a |
| 7 | 9L sparse last 2 | 86 | 3.34 | 1399 | 13.1MB | 2%/98% collapsed |
| 9 | 9L sparse aux=0.1 | 89 | 3.31 | 1415 | 13.1MB | 2%/98% collapsed |
| 11 | 9L Soft MoE last 2 | 189 | 3.25 | 636 | 14.6MB | balanced (dense) |
| full | 11L Soft MoE 8xH100 | 4704 | 1.1826 | 128 | 17.3MB | balanced |

Note: Tests 1-11 used 1 shard + 120s on 1xH100 for relative comparison.

### Key Insights
1. Sparse MoE router collapse is catastrophic at small scale — the
   feedback loop between routing and expert quality is too strong
2. torch.compile is critical for parameter golf step throughput —
   any architecture that breaks it pays a 2-3x speed penalty
3. Soft MoE sidesteps both problems but trades sparsity for density
4. Selective MoE (only deeper layers) is necessary to fit under 16MB

## Status
Work in progress. Final 10L run on 8xH100 pending to confirm
artifact fits under 16MB.

## How to Run

```bash
RUN_ID=soft_moe_10L \
MOE_MODE=soft \
NUM_EXPERTS=2 \
NUM_LAYERS=10 \
MOE_START_LAYER=8 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
