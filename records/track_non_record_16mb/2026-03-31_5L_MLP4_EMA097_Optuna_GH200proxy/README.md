# 5L MLP×4 EMA Optuna — Proxy Submission (GH200 MIG)

**Author:** pzydron
**Date:** 2026-03-30
**Track:** Non-record (proxy hardware — GH200 MIG 3g.48gb, not 8×H100)
**val_bpb (int6, stride=512):** 1.3622
**val_bpb (float proxy):** 1.3246
**Artifact size:** 12.5 MB (int6+zlib)

---

## What this submission demonstrates

This is a **proxy-scale submission** documenting an active research methodology prior to obtaining H100 compute for the full competition run. All experiments were conducted on owned hardware (8×V100 SXM2-32GB, GH200 MIG) at a 600-second budget (~820 steps).

## Key validated improvements over baseline

Starting from val_bpb 1.7747 (original baseline defaults):

| Technique | Δbpb | Validated |
|-----------|------|-----------|
| WARMDOWN_ITERS 3500→100 | −0.391 | ✅ |
| NUM_LAYERS optimum (5 for proxy) | −0.038 | ✅ |
| Optuna v1 optimizer params (TPE, 25 trials) | −0.006 | ✅ |
| MLP_MULT 3→4 | −0.009 | ✅ |
| VE_LAYERS dynamic fix (was pointing to layers 9,10 in a 5-layer model) | −0.004 | ✅ |
| EMA_DECAY 0.997→0.97 (calibrated for proxy run length) | −0.002 | ✅ |
| **Total improvement** | **−0.450** | |

## Architecture

```
vocab_size = 1024 (SP-1024 tokenizer)
num_layers = 5
model_dim  = 512
mlp_mult   = 4.0  (hidden = 2048)
num_heads  = 8
num_kv_heads = 4  (GQA)
bigram_vocab_size = 4096
bigram_dim = 1024
ve_layers = "3,4"  (last 2 layers, dynamic)
ema_decay = 0.97
warmdown_iters = 100
```

## Optimizer (Optuna v1 best, TPE sampler, 25 trials)

```
MUON_WD = 0.025
ADAM_WD = 0.0014
MUON_MOMENTUM = 0.947
MATRIX_LR = 0.068
SCALAR_LR = 0.042
GRAD_CLIP_NORM = 0.308
MUON_BETA2 = 0.986
MUON_MOMENTUM_WARMUP_STEPS = 1644
```

## Next steps (planned for H100 run)

Based on competitive analysis of all leaderboard submissions:
- Scale to 11 layers + MLP×3 (~22M params, fits 16 MB with int6+zstd-22)
- seq_len: 1024 → 2048
- EVAL_STRIDE: 512 → 64 (competition standard)
- EMA_DECAY: 0.97 → 0.995 (appropriate for 50k+ steps)
- LateQAT threshold: 0.30 → 0.15
- Add zstd-22 compression

Estimated val_bpb after H100 full run with target architecture: **~1.185 bpb**.

## Reproducibility

```bash
# Proxy run (GH200 MIG 3g.48gb, 600s)
MAX_WALLCLOCK_SECONDS=600 SKIP_INT6_EVAL=0 EVAL_STRIDE=64 \
  bash run_baseline.sh
```

Requires: `data/datasets/fineweb10B_sp1024`, `data/tokenizers/fineweb_1024_bpe.model`.
