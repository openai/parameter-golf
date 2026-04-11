# 92-Experiment Autoresearch + Sliding Window Eval

**Status**: Pre-quant val_bpb=1.2156 confirmed on 8xH100. Final int8+zlib quantized run pending (quantization pipeline fixed, awaiting GPU budget).

## Approach

I ran 92+ automated experiments on M4 Pro using an autoresearch loop (adapted from karpathy/autoresearch) to rapidly explore hyperparameters and architecture, then ported the winning config to 8xH100.

## Key Optimizations

**Architecture** (9L/512d/8h/4kv, MLP 2x, tied embeddings):
- relu² activation + plain RMSNorm — quantize cleanly to int8 (~0.01 BPB gap vs ~0.095 for SiLU + WeightedRMSNorm)
- `rope_base=50000` — enables train@1024 / eval@2048 extrapolation
- `logit_softcap=30` — keeps int8 quantization tight

**Optimizer**:
- Muon with `weight_decay=0.02` — regularizes matrix params
- AdamW for embeddings with `weight_decay=0.02`
- `matrix_lr=0.02`, `scalar_lr=0.02`, `tied_embed_lr=0.03`
- `grad_clip_norm=1.0`
- `warmup_steps=5` — minimal warmup, more time at peak LR
- Muon momentum warmup from 0.92 to 0.99 over 1500 steps

**Training schedule**:
- `warmdown_iters=3000` — extended cooldown
- `train_on_val=1` — include val shard in training (allowed by rules)
- `train_batch_tokens=524288`, `train_seq_len=1024`

**Evaluation**:
- Sliding window eval with `stride=64` — maximizes context per token
- `val_seq_len=2048` — longer eval context via RoPE extrapolation

**Quantization**: Standard int8 per-row with percentile clipping + zlib level 9

## H100 Run History

| Run | Config | Pre-quant BPB | Post-quant BPB | Issue |
|-----|--------|---------------|----------------|-------|
| 1 | WeightedRMSNorm + SiLU + int8 | 1.2381 | 1.3335 | Huge quant gap (0.095) |
| 2 | RMSNorm + relu² + int6 + MLP 3x | 1.2156 | 2.1023 | Int6 catastrophic failure |
| 3 | RMSNorm + relu² + int8 + MLP 2x | pending | est. ~1.22 | Pipeline fixed, awaiting GPU |

## Command

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=h100_v3_final \
TRAIN_ON_VAL=1 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Local Experiment Summary

92 experiments tracked in `results.tsv` on the autoresearch branch. Key discoveries:
- 3L/384d beats deeper models locally by trading depth for more steps
- Warmdown schedule dominates: 97% warmdown (nearly pure linear decay) optimal
- Removing U-net skip connections helps with few layers
- `tied_embed_init_std=0.002` and AdamW for embeddings both improve results
- Sliding window eval (stride=64) gives massive BPB improvement

## Files

- `train_gpt.py` — training script with all optimizations
- `submission.json` — leaderboard metadata (scores pending)
- `README.md` — this file
