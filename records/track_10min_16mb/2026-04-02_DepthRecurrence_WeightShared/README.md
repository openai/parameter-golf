# [WIP] Depth Recurrence via Weight-Shared Transformer Blocks

**Status: In Progress** | Target: < 16 MB | 8xH100 SXM, 600s

## Approach

Weight-shared depth recurrence: instead of 11 unique transformer blocks, share weights across a smaller set of blocks and iterate multiple times, achieving 20+ effective layers within the same 16MB parameter budget.

This technique is listed as an OpenAI "Request for PR" and has not been successfully demonstrated in any submission to date.

## Core Idea

Current SOTA (PR #1019, 1.1147 BPB) allocates ~4M parameters per layer across 11 unique blocks. With weight sharing:

- **4 shared blocks x 5 iterations = 20 effective layers**
- Freed parameter budget reallocated to wider dimensions, larger BigramHash, or additional architectural capacity
- Per-layer conditioning via layer index embeddings and learned scalar gates ensures each iteration is distinct
- Compatible with existing GPTQ int6 quantization (shared weights quantized once, applied K times)

## Architecture Plan

Building on the PR #1019 stack (GPTQ + XSA + BigramHash + Parallel Muon):

| Component | Change |
|-----------|--------|
| Layer structure | K shared blocks with N iterations (K*N effective depth) |
| Per-iteration conditioning | Layer index embeddings + learned gates |
| Normalization | Per-iteration RMSNorm to stabilize deep recurrence |
| Skip connections | Adapted U-Net skips for recurrent structure |
| Remaining stack | XSA, BigramHash, SmearGate, Partial RoPE, VE128 unchanged |

## Why This Should Work

1. Scaling laws show depth is more parameter-efficient than width at fixed budget
2. Universal Transformer (Dehghani et al.) demonstrated weight sharing matches standard transformers with fewer parameters
3. Quantization gains are hitting diminishing returns (int6 to int5); the next improvement likely comes from structural parameter reallocation
4. Extra forward passes fit within the 10-minute compute budget and 10-minute eval budget

## Development Setup

- Local experimentation: DGX Sparks (Blackwell GPU, 128GB unified memory)
- Validation: 8xH100 SXM via RunPod

## Lineage

```
PR #1019 (Current SOTA, 1.1147 BPB)
    +-- This work adds:
        +-- Weight-shared depth recurrence (K blocks x N iterations)
        +-- Per-layer conditioning (index embeddings, learned gates)
        +-- Adapted skip connections for recurrent structure
```

## Author

GitGeeks (milhouse)
