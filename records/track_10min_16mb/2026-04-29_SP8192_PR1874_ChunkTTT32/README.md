# SP8192 PR #1874 + Optimized Hyperparameters — val_bpb 1.06844 (3-seed mean)

## Results

| Seed | Pre-quant BPB | Post-quant BPB | **Post-TTT BPB** | Artifact (bytes) | Train time | Eval time |
|------|---------------|----------------|------------------|-------------------|------------|-----------|
| 1337 | 1.06960       | 1.07925        | **1.06798**      | 15,950,405        | 599.64s    | 409.6s    |
| 42   | 1.06984       | 1.07948        | **1.06824**      | 15,952,215        | 599.65s    | 421.0s    |
| 2025 | 1.07060       | 1.08028        | **1.06909**      | 15,948,755        | 599.56s    | 381.2s    |
| **Mean** | **1.07001** | **1.07967** | **1.06844**  | **15,950,458**    | **599.62s** | **403.9s** |
| **Std** | 0.00053    | 0.00053        | **0.00058**      | 1,730             | 0.05s      | 20.2s     |

## Configuration

- **Base code:** PR #1874 (AjAnubolu) verbatim — no code modifications
- **Environment variables:** `MIN_LR=0.10 QK_GAIN_INIT=5.25 GATE_ATTN_WIDTH=24 GPTQ_RESERVE_SECONDS=0.5 VAL_LOSS_EVERY=0`
- **Hardware:** 8xH100 80GB SXM (RunPod on-demand)
- **Data template:** `c5dbhtfrrt` (SP8192, 128 train + 1 val shards)

## Techniques (all from PR #1874, activated via env vars)

1. **LQER Asymmetric Rank-4** — SVD-based low-rank quantization error reduction on top-K=3 highest-error GPTQ residuals
2. **SmearGate + Attention Output Gate (width 24)** — per-layer smoothing + full-dim attention gating
3. **Polar Express Newton-Schulz** — 5 per-iteration minimax-tuned coefficient tuples for Muon optimizer
4. **MIN_LR=0.10** — warmdown LR floor at 10% of max (prevents LR collapse to zero)
5. **QK_GAIN_INIT=5.25** — per-head query-key attention scaling
6. **GATE_ATTN_WIDTH=24** — doubled attention gate capacity
7. **GPTQ_RESERVE_SECONDS=0.5** — maximizes training steps (default 4.0 wastes ~28 steps)
8. **VAL_LOSS_EVERY=0** — eliminates mid-training eval overhead (~14s saved = ~112 extra steps)
9. **Phased Score-First TTT** — 3-phase AdamW LoRA-TTT (rank 128), score-first ordering

## Rule Compliance

- Score-first phased TTT (no re-scoring)
- No pre-quant TTT on validation data
- No n-gram cache or PPM
- No CaseOps, no casefold — standard SP8192 UTF-8 byte counting
- Artifact < 16,000,000 bytes (max 15,952,215 B)
- Train time < 600s (max 599.65s), eval time < 600s (max 421.0s)

## How to reproduce

```bash
SEED=1337 MIN_LR=0.10 QK_GAIN_INIT=5.25 GATE_ATTN_WIDTH=24 \
  GPTQ_RESERVE_SECONDS=0.5 VAL_LOSS_EVERY=0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Attribution

Built entirely on PR #1874 (AjAnubolu), which itself builds on PR #1790 (miaoyuxun), PR #1344 (Polar Express), PR #1787 (nprime06), PR #1797 (dexhunter).
