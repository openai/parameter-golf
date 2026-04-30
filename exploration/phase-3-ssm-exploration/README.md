# Phase 3: SSM Exploration (Mamba / Griffin)

**Dates:** Mar 23-24, 2026
**Goal:** Replace attention with state-space models (Mamba-3, Griffin RG-LRU) for O(L) inference and higher throughput.
**Outcome:** Dead end under the 10-minute constraint. SSMs needed more training tokens to converge than attention, negating the throughput advantage.

## Runs

| Run | Architecture | Notes |
|-----|-------------|-------|
| 010-mamba3-n8-blk2 | Mamba-3, n=8, 2 blocks | First SSM attempt |
| 011-mamba3-n4-baseline | Mamba-3, n=4 | Reduced depth |
| 012-griffin-rglru-n4 | Griffin RG-LRU, n=4 | Alternative SSM |
| 001-griffin-n8-cpb-1gpu | Griffin n=8, 1 GPU | Compute-per-byte optimization |
| 002-griffin-n8-memfix-1gpu | Griffin memory fix | OOM debugging |
| 003-griffin-n8-memfix-4gpu | Griffin, 4 GPU | Scaling test |
| 004-griffin-n8-variants-8gpu | Griffin variants, 8 GPU | Full-scale comparison |
| 004-griffin-variants-DFS | Griffin + DFS | Combined approach |

## Key Findings

- SSMs are faster per step but converge slower per token
- Under 10-minute wall clock, the convergence disadvantage outweighed the throughput advantage
- Griffin was more stable than Mamba-3 but still couldn't match attention quality at matched training time
- Memory management was tricky with recurrent states + gradient checkpointing

## What Led to Phase 4

Pure SSM replacement didn't work. Phase 4 explored hybrid approaches: using depth-first search ordering, token injection, and prefix state prefilling to get more information into fewer steps.
