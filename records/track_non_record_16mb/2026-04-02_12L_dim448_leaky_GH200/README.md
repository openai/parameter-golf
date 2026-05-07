# 12L dim=448 LeakyReLU² BGVOCAB=2048 — Proxy Submission (GH200 MIG 1g.24gb)

**Author:** pzydron
**Date:** 2026-04-02
**Track:** Non-record (proxy hardware — GH200 MIG 1g.24gb, not 8×H100)
**val_bpb (int6+zstd-22, stride=64):** 1.23263493
**Artifact size:** 15.58 MB (int6+zstd-22)

---

## What this submission demonstrates

Proxy-scale submission (5000 steps on GH200 MIG 1g.24gb = 1/7 of an H100) documenting validated
architectural improvements over the baseline. All results use **EVAL_STRIDE=64** (competition standard).

## Key improvements over baseline

Starting from val_bpb 1.7747 (original baseline defaults), validated at 5k steps:

| Technique | Δbpb | Validated |
|-----------|------|-----------|
| WARMDOWN_ITERS 3500→500 | −0.391 | ✅ |
| NUM_LAYERS 3→12 (proxy optimum=5, full-scale optimum=12) | −0.062+ | ✅ |
| Optuna v1 optimizer params (TPE, 25 trials) | −0.006 | ✅ |
| MLP_MULT 3→3 + dim 512→448 (budget fit) | 0 | budget fix |
| VE_LAYERS dynamic fix | −0.004 | ✅ |
| EMA_DECAY 0.997→0.995 (calibrated for full-scale) | stable | ✅ |
| LeakyReLU² (slope=0.1) vs ReLU² | −0.001 | ✅ |
| BIGRAM_VOCAB_SIZE 4096→2048 (budget fix) | −0.002 | budget fix |

## Architecture

```
vocab_size         = 1024   (SP-1024 tokenizer)
num_layers         = 12
model_dim          = 448
mlp_mult           = 3.0    (hidden = 1344)
num_heads          = 8
num_kv_heads       = 4      (GQA 2:1)
bigram_vocab_size  = 2048
bigram_dim         = 1024
xsa_last_n         = 4      (XSA on last 4 layers)
ve_layers          = "10,11" (last 2 layers, dynamic)
ema_decay          = 0.995
warmdown_iters     = 500
activation         = LeakyReLU² (F.leaky_relu(x, 0.1).square())
quantization       = int6 + zstd-22
```

## Optimizer (Optuna v1 best, TPE sampler, 25 trials)

```
MUON_WD                     = 0.025
ADAM_WD                     = 0.0014
MUON_MOMENTUM               = 0.947
MATRIX_LR                   = 0.068
SCALAR_LR                   = 0.042
GRAD_CLIP_NORM              = 0.308
MUON_BETA2                  = 0.986
MUON_MOMENTUM_WARMUP_STEPS  = 1644
```

## Reproducibility

```bash
# Proxy run (GH200 MIG 1g.24gb, 5000 steps, ~90 min)
MAX_WALLCLOCK_SECONDS=36000 ITERATIONS=5000 EVAL_STRIDE=64 \
  TRAIN_BATCH_TOKENS=262144 \
  python train_gpt.py
```

Requires: `data/datasets/fineweb10B_sp1024`, `data/tokenizers/fineweb_1024_bpe.model`.

## Note on proxy vs full-scale

Proxy scale (5k steps) underestimates final quality. All top-10 leaderboard entries use 12+ layers
trained for 20k+ steps on 8×H100. This submission is a stepping stone to a full H100 run.
