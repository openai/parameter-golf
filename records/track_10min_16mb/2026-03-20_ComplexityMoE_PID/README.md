# Partitioned MoE + PID + BigramHash + Int5/Int6 + SlidingWindow

**Author:** Boris Peyriguere (Complexity-ML)
**Date:** 2026-03-20
**Score:** Pending (awaiting compute credits)

---

## Summary

Novel submission combining **partition-based isolation** (inspired by `sha256(api_key:user_id) % 64` security partitioning) with proven leaderboard techniques to push BPB as low as possible under the 16MB / 10min constraint.

- **Per-layer hash routing** — `(token_id × 36313) ⊕ (layer_id × 27191) % E`
  Deterministic, zero-overhead. Each layer routes tokens to different experts, breaking co-activation bias vs flat `token_id % E`. No two layers share the same token-expert affinity.

- **Layer budget partitioning** — 3 tiers
  P0 (input: 2 experts, MLP 2×) · P1 (middle: 4 experts, MLP 3×) · P2 (output: 4 experts, MLP 2×)
  Parameters allocated where marginal BPB gain is highest.

- **INL BetaMu PID dynamics**
  Error-gated causal conv1d with learnable equilibrium μ on middle layers (3–5).
  Classical GQA with RoPE on input/output (0–2, 6–8).

- **BigramHash(10240) + SmearGate**
  Hash consecutive token pairs into a 10240-bucket learned embedding (dim=128) + previous-token gating.

- **Int5/Int6 mixed quantization**
  Int5 [-16, 15] for MLP (1.88× zstd ratio) · Int6 [-32, 31] for attention · FP16 for tied embeddings.
  3% magnitude pruning before compression.

- **Sliding window eval** (stride=64, window=2048)
  Extracted to standalone `eval_sliding.py`.

- **SWA** start_frac=0.4 (last 40% warmdown, every 50 steps)
  **Muon momentum** 0.99 · **WD** 0.04 · **seq_len** 2048 · **model_dim** 512 · **batch** 786K tokens

- **LoRA TTT** — rank-8 per-document test-time training at eval

---

## Architecture

```
Embedding + BigramHash(10240) → RMSNorm → SmearGate → [Block × 9] → FinalNorm → LM Head

P0 (layers 0-2):  GQA (RoPE),     2 experts, MLP 2×   ← input partition
P1 (layers 3-5):  INL BetaMu,     4 experts, MLP 3×   ← middle partition (max capacity)
P2 (layers 6-8):  GQA (RoPE),     4 experts, MLP 2×   ← output partition

U-Net skip connections · Sort-and-split MoE dispatch · fullgraph-safe
```

---

## Files

| File | Description |
|------|-------------|
| `train_gpt.py` | Training script (1435 lines, under 1500 limit) |
| `eval_sliding.py` | Sliding window eval (standalone or imported by train) |
| `config.json` | All hyperparameters |
| `i64_moe_kernel.cu` | Optional CUDA kernel for MoE dispatch |
| `submission.json` | Submission metadata |

---

## How to Run

```bash
# Download data
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train (8xH100)
RUN_ID=partition_v2 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_ComplexityMoE_PID/train_gpt.py

# Standalone sliding window eval
MODEL_PATH=final_model.int8.ptz python3 \
  records/track_10min_16mb/2026-03-20_ComplexityMoE_PID/eval_sliding.py
```

---

## Test Plan

- [ ] Verify training completes in <10min on 8xH100
- [ ] Confirm artifact size <16MB after int5/int6+zstd-22
- [ ] Run 3 seeds, report mean±std (p < 0.01)
- [ ] Validate val_bpb with sliding window eval (stride=64)
- [ ] Verify quantization roundtrip integrity

---

## Status

Awaiting RunPod compute credits to produce training logs and final val_bpb score.
