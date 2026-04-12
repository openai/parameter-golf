# JEPA-NTP: Auxiliary Latent Losses for Next-Token Prediction

**Result: Negative.** JEPA-style auxiliary losses do not improve next-token prediction in the parameter golf regime.

**Best JEPA variant post-quant val_bpb: 1.4352** (baseline: **1.4326**)

## Motivation

Can we improve language model representations by borrowing the two-loss recipe from LeCun's LeWorldModel (LeWM) paper? LeWM uses:
1. **Spectral variance floor** — prevents dimensional collapse in hidden states
2. **Cosine-MSE latent prediction** — predicts the next-position hidden state via a dedicated MLP head

We adapted both losses for autoregressive language models and tested whether they improve `val_bpb` in the parameter golf setting (9-layer, 512-dim, ~17M param transformer).

## The Compound Loss

```
L = L_CE + alpha * L_cosine_mse(h_hat_{t+1}, sg(h_{t+1})) + lambda * L_spectral(delta_h_t)
```

| Component | What it does | Hyperparameter |
|-----------|-------------|----------------|
| `L_CE` | Standard cross-entropy NTP | — |
| `L_cosine_mse` | Predicts next-position latent via 2-layer MLP | alpha = 0.1 |
| `L_spectral` | Penalizes eigenvalues of Cov(delta_h) below epsilon | lambda = 0.01, eps = 0.01 |

Key adaptations for language (vs. LeWM's vision setting):
- Applied to hidden-state **deltas** (delta_h = h_{t+1} - h_t), not absolute states
- Only penalizes total dimensional collapse, not full isotropy (SIGReg would destroy language embedding structure)
- L2-normalize before MSE (cosine distance — avoids magnitude conflicts with CE)
- Stop-gradient on target prevents representational shortcuts

## Hardware

- 2x NVIDIA RTX PRO 6000 Blackwell (98 GB VRAM each)
- RunPod cloud instance
- 1 epoch of FineWeb-10B SP1024 (~1B tokens, 477 steps at 2M tokens/step)

## Results

All runs: torch.compile enabled, 1 epoch, identical data/hyperparameters.

### JEPA Experiments (train_jepa_ntp.py)

| Experiment | Config | val_bpb (pre-quant) | val_bpb (post-quant) | Quant gap | tok/s | Int8+zlib bytes |
|-----------|--------|:-------------------:|:--------------------:|:---------:|------:|----------------:|
| **Baseline** | L_CE only | **1.4152** | **1.4326** | 0.0174 | 2,119,557 | 9,900,240 |
| Exp4 JEPA targeted | L_CE + spectral + cosine_mse (layers 2-5) | 1.4170 | 1.4352 | 0.0182 | 1,702,648 | 9,903,685 |

### Architectural Experiments (train_modded.py)

| Experiment | Config | Params | val_bpb (pre-quant) | val_bpb (post-quant) | tok/s | Int8+zlib bytes |
|-----------|--------|-------:|:-------------------:|:--------------------:|------:|----------------:|
| **Baseline** | Stock 9L/512d/4KV | 17.06M | **1.4152** | **1.4326** | 2,119,557 | 9,900,240 |
| MQA + Value Embeds | 1 KV head + VE on first/last 2 layers | 15.42M | 1.4277 | 1.4439 | 2,201,000 | 9,680,000 |
| MQA + VE + 3x MLP | Above + 3x MLP multiplier | 20.14M | 1.4190 | 1.4364 | 1,936,000 | 12,120,000 |

### Verdict

The baseline wins on every metric. At 1 epoch of training:
- JEPA: +0.0018 BPB worse, 20% slower throughput
- MQA+VE: +0.012 BPB worse (dropping from 4 to 1 KV head loses too much attention capacity)
- MQA+VE+3xMLP: +0.004 BPB worse (3x MLP partially recovers but doesn't justify the size increase)

## Key Findings

### 1. JEPA auxiliary losses are orthogonal to quantization

The spectral floor and cosine-MSE losses operate on **hidden-state geometry** (eigenvalue spectrum, latent prediction), but int8 quantization operates on **weight distributions** (per-row clipping, rounding error). These are nearly independent — JEPA cannot improve quantization robustness.

### 2. Spectral floor works mechanically but doesn't help language modeling

The spectral loss successfully prevents dimensional collapse:
- Effective rank improved from 424 to 445 out of 512 dimensions (83% -> 87%)
- Tail singular values (sv_058-063) stayed at ~2,300 (far from zero)
- The loss spiked to 0.045 at step 25 (catching early collapse), then decayed to 0 by step 175

But this improved dimension utilization didn't translate to better next-token prediction at 17M params. The model is too small for dimensional collapse to be the binding constraint.

### 3. Cosine-MSE predictor was essentially inert

The predictor MLP's loss values were tiny throughout training (0.001-0.003). With alpha=0.1, the actual gradient contribution was negligible. The predictor added ~1M parameters (not counted in submission size) and 20% throughput overhead for no benefit.

### 4. Critical methodology lesson: torch.compile confound

An initial apparent 0.02 BPB improvement from JEPA was traced to a **torch.compile confound** — the baseline ran uncompiled while JEPA ran compiled. Once both used torch.compile:

| Run | Compile | val_bpb |
|-----|---------|--------:|
| Baseline (uncompiled) | off | 1.9496 |
| JEPA exp4 (compiled) | on | 1.9065 |
| **Baseline (compiled)** | **on** | **1.8990** |

torch.compile alone provides ~0.05 BPB improvement beyond just speed — it affects optimization quality (likely via reduced numerical noise in fused kernels).

### 5. MQA is harmful at this scale

Going from 4 to 1 KV head in a 9-layer model loses too much attention capacity. Value embeddings (zero-compute-cost per-token V lookups shared between first/last layers) don't compensate. Even adding a 3x MLP multiplier to use the freed parameters doesn't fully recover quality.

## Experiment Framework

The submission includes a reusable experimental framework:

```
jepa_ntp/
  config.py                  -- Experiment configurations (baseline, exp1-4)
  train_jepa_ntp.py          -- JEPA training script (wraps train_gpt.py)
  train_modded.py            -- MQA + Value Embeddings script
  losses/
    spectral_floor.py        -- Spectral variance floor loss
    cosine_mse.py            -- Cosine-MSE prediction loss + LatentPredictor MLP
  metrics/
    effective_rank.py         -- Effective rank diagnostic
    singular_spectrum.py      -- Full SV spectrum logging
    latent_smoothness.py      -- Latent path curvature + cosine smoothness
```

### Running

```bash
# JEPA experiment (exp4 = best JEPA variant)
EXPERIMENT=exp4_targeted RUN_ID=exp4_run1 \
    torchrun --standalone --nproc_per_node=2 jepa_ntp/train_jepa_ntp.py

# Modded architecture experiment
EXPERIMENT=exp5_modded RUN_ID=modded_run1 \
    torchrun --standalone --nproc_per_node=2 jepa_ntp/train_modded.py
```

### WandB Diagnostics

The framework logs rich diagnostics to WandB:
- `diagnostics/effective_rank_delta` — primary anti-collapse metric (should stay near model_dim)
- `spectrum/sv_*` — full singular value spectrum (smooth decay = healthy, cliff to zero = collapse)
- `loss/cosine_mse` — predictor loss (should decrease if predictor is learning)
- `diagnostics/curvature` — latent path curvature (decreasing = path straightening)
- `diagnostics/cosine_smoothness` — consecutive velocity cosine similarity

## Design Decisions

1. **Inline hidden state capture** (no hooks) — hooks break torch.compile. Instead, JEPAGPT subclasses GPT and returns captured hidden states from forward(). When capture_layers=None, branches are dead-code-eliminated by the compiler.

2. **DDP before compile** — PyTorch requires DDP wrapping before torch.compile. The predictor MLP also needs DDP wrapping for weight synchronization across ranks.

3. **Predictor weights excluded from submission** — the LatentPredictor MLP is only used during training. The serialized model is identical to baseline (same architecture, same param count).

## What Would Be Worth Trying Next

Based on leaderboard analysis, the winning approaches use fundamentally different strategies:
1. **More training** — 3.7 epochs (not 1) to match competition baseline
2. **QAT (quantization-aware training)** — fake int6 quantization with STE during warmdown
3. **Better compression** — zstd-22 instead of zlib, int6 instead of int8
4. **More layers** — 11 instead of 9 (all top entries use deeper models)

These are orthogonal to auxiliary training losses and address the actual bottlenecks.

## References

- [LeWorldModel: Training JEPA Models with Two Loss Terms](https://arxiv.org/abs/2503.XXXXX) — Garrido, Bordes, Assran, Ballas, Bardes, Najman, LeCun (2026)
- [SIGReg: Sketched Isotropic Gaussian Regularizer](https://arxiv.org/abs/2308.11809)
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — Value Embeddings, MQA tricks
- [OpenAI Parameter Golf](https://github.com/openai/parameter-golf)
