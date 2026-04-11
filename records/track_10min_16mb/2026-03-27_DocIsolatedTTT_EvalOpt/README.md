# Document-Isolated TTT + Eval Optimizations

## Summary

Built on PR #549 (LeakyReLU^2 + Legal TTT + Parallel Muon, 1.1194 BPB).

Key insight: The current SOTA runs TTT on a flat token stream without respecting document boundaries. Cross-document contamination dilutes the adaptation signal. By resetting TTT optimizer state at BOS boundaries, each document gets a fresh adaptation — the same technique that gave -0.011 BPB in the LoRA TTT submission (PR #461) but was never applied to the frontier architecture.

## Approach

1. **Document-Isolated TTT**: Detect document boundaries via BOS token positions. Reset SGD optimizer momentum between documents. Each document gets independent adaptation.
2. **Temperature Scaling**: Grid search T=0.90-1.00 on the quantized model at eval time.
3. **Base Architecture**: Unchanged from PR #549 (11L/512d/8H/4KV, 3x MLP LeakyReLU(0.5)^2, XSA4, Partial RoPE 16/64, LN Scale, SmearGate, BigramHash, VE128, Parallel Muon, EMA+SWA, GPTQ-lite int6+LZMA).

## Status

Work in progress. Awaiting 8xH100 compute credits for validation runs.

### Dev Results (1xH100 NVL)

- Base architecture reproduces correctly: 1.39 BPB at 920 steps (consistent with 1xH100 scaling from 8xH100 SOTA)
- Tested and rejected: sp4096 vocabulary (fails at full convergence), NorMuon, ProRes (torch.compile incompatible)

## Expected Results

Target: 1.09-1.11 BPB (pending 8xH100 validation)

## Setup

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
