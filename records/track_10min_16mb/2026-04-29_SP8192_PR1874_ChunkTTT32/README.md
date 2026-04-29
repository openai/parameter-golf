# SP8192 PR #1874 + TTT Chunk=32 — val_bpb 1.06990 (3-seed mean)

## Results

| Seed | Pre-quant BPB | Post-quant BPB | **Post-TTT BPB** | Artifact (bytes) | Train time | Eval time |
|------|---------------|----------------|------------------|-------------------|------------|-----------|
| 1337 | 1.07132       | 1.08122        | **1.06985**      | 15,943,571        | 596.06s    | 595.5s    |
| 42   | 1.07169       | 1.08151        | **1.07017**      | 15,950,196        | 596.12s    | 560.8s    |
| 2025 | 1.07134       | 1.08107        | **1.06968**      | 15,946,736        | 596.11s    | 556.1s    |
| **Mean** | **1.07145** | **1.08127** | **1.06990**  | **15,946,834**    | **596.10s** | **570.8s** |
| **Std** | 0.00021    | 0.00022        | **0.00025**      | 3,314             | 0.03s      | 21.5s     |

## Configuration

- **Base code:** PR #1874 (AjAnubolu) verbatim
- **Environment variable:** `TTT_CHUNK_SIZE=32` (default is 48)
- **Hardware:** 8xH100 80GB SXM (RunPod on-demand)
- **Data template:** `c5dbhtfrrt` (SP8192, 128 train + 1 val shards)

## Techniques (all from PR #1874)

1. **LQER Asymmetric Rank-4** — SVD-based low-rank quantization error reduction on top-K=3 highest-error GPTQ residuals
2. **SmearGate + Attention Output Gate (width 24)** — per-layer smoothing + full-dim attention gating
3. **Polar Express Newton-Schulz** — 5 per-iteration minimax-tuned coefficient tuples for Muon optimizer
4. **MIN_LR=0.10** — warmdown LR floor at 10% of max
5. **Phased Score-First TTT** — 3-phase AdamW LoRA-TTT (rank 128), score-first ordering
6. **TTT_CHUNK_SIZE=32** — smaller chunks = more gradient updates per document during TTT eval (our addition)

## Rule Compliance

- Score-first phased TTT (no re-scoring)
- No pre-quant TTT on validation data
- No n-gram cache or PPM
- No CaseOps, no casefold — standard SP8192 UTF-8 byte counting
- Artifact < 16,000,000 bytes (max 15,950,196 B)
- Train time < 600s, eval time < 600s

## How to reproduce

```bash
# Seeds 1337, 42, 2025
SEED=1337 TTT_CHUNK_SIZE=32 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Attribution

Built entirely on PR #1874 (AjAnubolu), which itself builds on PR #1790 (miaoyuxun), PR #1344 (Polar Express), PR #1787 (nprime06), PR #1797 (dexhunter).
