# Hyperparameter-Tuned KV2 + FP16 Embed

**val_bpb: TBD** (pending 8xH100 SXM validation)

## Summary

Systematic hyperparameter optimization of the baseline architecture, plus FP16 embedding export. Seven throughput-neutral changes that compound to measurable improvement without altering the core model architecture.

## Changes from Baseline

| Parameter | Baseline | Ours | Rationale |
|-----------|----------|------|-----------|
| NUM_KV_HEADS | 4 | **2** | GQA 8:2 saves ~1.2M params, throughput-positive |
| MATRIX_LR | 0.04 | **0.048** | Higher Muon LR for KV2. Swept [0.036-0.052] |
| WARMDOWN_ITERS | 1200 | **600** | Shorter warmdown preserves high LR longer. Swept [400-1200] |
| SCALAR_LR | 0.04 | **0.03** | Lower Adam LR for control params. Swept [0.03-0.06] |
| MUON_MOMENTUM | 0.95 | **0.97** | Higher momentum. Swept [0.93-0.99] |
| QK_GAIN_INIT | 1.5 | **1.35** | Lower initial attention sharpness. Swept [1.0-2.0] |
| tok_emb export | int8 | **fp16** | Avoids quant noise compounding through input+output |

## Key Insight

In this wallclock-capped competition, **throughput is everything**. We systematically tried SwiGLU, width reinvestment, depth recurrence, cosine warmdown, and shorter sequence length — all were net negative because throughput loss outweighed per-step quality gains. The winning strategy was throughput-neutral hyperparameter optimization.

## What We Tried and Rejected

| Change | Why it failed |
|--------|--------------|
| SwiGLU activation | 2.2x slower per step under torch.compile |
| Width reinvestment (dim=528) | Slower per step, net negative |
| Depth recurrence (shared blocks) | Underpowered at dim=512; dim=768 too slow |
| 10 layers (up from 9) | Training instability on 8xH100 |
| Cosine warmdown | Worse than linear |
| MUON_BACKEND_STEPS=3 | Quality drop too large |
| TRAIN_SEQ_LEN=512 | Model needs full 1024 context |

## Preliminary Results

1xH100 (10 min, 1 GPU): val_bpb = 1.3003 (baseline: 1.3106, -0.8%)

Pending 8xH100 SXM validation with full batch size.

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All defaults baked in. No env overrides needed.

## Files

- `train_gpt.py` — modified training script (based on latest upstream)
- `submission.json` — metadata (val_bpb TBD)
- `README.md` — this file
