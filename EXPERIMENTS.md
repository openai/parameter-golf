# Parameter Golf — Experiment Tracker

## Goal
Beat the baseline **1.2244 BPB** (post-int8-quantization) on FineWeb validation, within 16MB / 10 min on 8×H100.

## Current Status
- **Infrastructure**: 1×H100 SXM on RunPod ($2.69/hr), pipelined autoresearch loop
- **Baseline (1-GPU, 3 min)**: 1.6419 BPB — our iteration benchmark
- **Baseline (8-GPU, 10 min)**: 1.2244 BPB — official submission benchmark
- **Best so far**: **1.5956 BPB** (12L, 512d, MLP 3x — using 14.7MB of 16MB budget)

## Experiment Log

| # | Change | BPB | vs Best | Size (bytes) | Kept? | Notes |
|---|--------|-----|---------|--------------|-------|-------|
| 0 | Baseline (9L, 512d, MLP 2x, 4KV, tied emb) | 1.6419 | — | 8,899,669 | ✓ | Official config, 334 steps in 3 min |
| 1 | SiLU activation instead of ReLU² | 1.6968 | +0.055 | 8,835,877 | ✗ | Worse — ReLU² better for this regime |
| 2 | 11 layers × 448 dim (vs 9×512) | 1.6201 | -0.022 | 8,769,774 | ✓ | Deeper/narrower helps |
| 99 | 12L × 512d, MLP 3x (wider MLP) | 1.5956 | -0.025 | 14,759,375 | ✓ | **Best so far.** Uses more of size budget |
| 3 | matrix_lr 0.05 → 0.08 | 1.5698 | -0.026 | 16,575,250 | ✗ | Great BPB but **over 16MB**! |
| 100+ | (autoresearch running with Opus, pipelined) | — | — | — | — | — |

## Ideas to Try

### High Priority (likely to help)
- [ ] Learning rate sweep — matrix_lr 0.08 showed big gains but went over size; try 0.06-0.07
- [x] More layers at same param count (depth vs width tradeoff) → **helped!** (exp #2, #99)
- [x] MLP expansion ratio tuning → **3x better than 2x** (exp #99)
- [ ] Sequence length changes (shorter = more steps in time budget)
- [ ] Weight sharing / layer tying (effective depth for free)
- [ ] Slightly smaller model + higher LR (stay under 16MB while getting the LR benefit)

### Medium Priority
- [ ] Different attention configs (more/fewer heads, different KV head counts)
- [ ] Warmdown schedule tuning
- [ ] Batch size adjustments
- [ ] Gradient clipping
- [ ] Logit softcap value

### Speculative / High Risk
- [ ] Depth recurrence (run same layers multiple times)
- [ ] Mixture of experts
- [ ] Quantization-aware training
- [ ] Sequence length warmup (start short → go long)
- [ ] Alternative optimizers or Muon hyperparameters

## What We've Learned
- ReLU² outperforms SiLU in this small-model/short-training regime (exp #1)
- Deeper models help: 9→11→12 layers each improved BPB (exp #2, #99)
- Wider MLP (3x vs 2x) helps significantly (exp #99)
- Higher matrix_lr (0.08) gives big BPB gains but needs a smaller model to fit 16MB (exp #3)
- Current best (12L/512d/MLP3x) uses 14.7MB — close to the 16MB limit
- On 1×H100, step time is ~540ms (vs ~44ms on 8×H100), ~334 steps in 3 min
- Pipelined proposals: speculative next-experiment while training runs

## Infrastructure Notes
- Pod: RunPod secure cloud, 1×H100 SXM 80GB, pod ID `dfmh6pu2s39twj`
- SSH: `ssh root@216.243.220.216 -p 15584`
- Auth: Claude Max subscription (OAuth)
- Autoresearch output: `tail -f /workspace/parameter-golf/autoresearch.out`
- History: `/workspace/parameter-golf/autoresearch/history.jsonl`
- Logs: `/workspace/parameter-golf/autoresearch/logs/`
- Experiment snapshots: `/workspace/parameter-golf/autoresearch/experiments/NNNN/`
- Pull results: `./pod.sh pull`
