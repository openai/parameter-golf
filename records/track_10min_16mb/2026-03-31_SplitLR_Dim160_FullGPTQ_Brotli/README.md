# Record: Split-LR + BigramHash(2816x160) + Full Hessian GPTQ + Brotli-11

**val_bpb: 1.1105** (3-seed mean, std 0.0006) | **1.8751 nats** | **~15.81 MB** | 8×H100 SXM, 600s | No TTT

Built on [PR #1019](https://github.com/openai/parameter-golf/pull/1019) by @abaybektursun (our previous merged SOTA).
Previous: PR #549 (1.1194) -> PR #1019 (1.1147) -> this (1.1105).

## Changes from PR #1019

| | PR #1019 | This |
|---|---|---|
| val_bpb | 1.1147 | **1.1105** |
| BigramHash | 3072 x 112 | **2816 x 160** (wider projection) |
| Compression | LZMA preset=9 | **Brotli quality=11 + byte-shuffle** |
| Code | 101,850 bytes | **23,349 bytes (minified)** |
| Split-LR | No | **Yes** (early=0.025, late=0.030) |
| Skip connections | Standard U-Net | **Sigmoid-gated U-Net** |
| QAT | Standard STE | **Soft-round (alpha ramp 1->16)** |
| GPTQ reserve | 14s | **9s** (more training time) |

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, no TTT)

| Seed | Steps | ms/step | Post-EMA BPB | **Sliding BPB** | val_loss (nats) | Artifact |
|------|-------|---------|--------------|-----------------|-----------------|----------|
| 1337 | 6,702 | 88.2 | 1.1314 | **1.1110** | 1.8758 | 15,807,723 |
| 42 | 6,708 | 88.1 | 1.1304 | **1.1098** | 1.8739 | 15,811,712 |
| 2025 | 6,712 | 88.1 | 1.1312 | **1.1108** | 1.8755 | 15,800,500 |
| **Mean** | **6,707** | **88.1** | | **1.1105** | **1.8751** | |

### Supplemental Diagnostics

| Seed | Post-EMA BPB | Roundtrip BPB | Sliding BPB | val_loss (nats) | Code size | Total submission | Train time | Eval time |
|------|--------------|---------------|-------------|-----------------|-----------|------------------|------------|-----------|
| 1337 | 1.1314 | 1.1346 | 1.1110 | 1.87584 | 23,349 | 15,807,723 | 591s | 86s |
| 42 | 1.1304 | 1.1335 | 1.1098 | 1.87387 | 23,349 | 15,811,712 | 591s | 87s |
| 2025 | 1.1312 | 1.1344 | 1.1108 | 1.87555 | 23,349 | 15,800,500 | 591s | 86s |
| **Mean** | **1.1310** | **1.1342** | **1.1105** | **1.87508** | **23,349** | | **591s** | **~86s** |

Current merged SOTA (PR #1019, exact 3-seed mean): **1.11473509 BPB** (**1.88217853 nats**).
This run's exact 3-seed mean: **1.11053346 BPB** (**1.87508426 nats**).
Delta: **-0.00709427 nats** (**-0.00420163 BPB**).

### Timing Budget

| Phase | Time |
|-------|------|
| Training (wallclock cap) | 591s |
| GPTQ calibration (reserved) | 9s |
| Post-EMA eval | ~2s |
| Int6 roundtrip eval | ~6s |
| Sliding window eval (stride=64) | ~78s |
| **Total eval** | **~86s** |

---

## What's New

### 1. Split-LR (Differential Learning Rates)

Different learning rates for early vs late transformer layers:
- Early layers (0-4): `MATRIX_LR_EARLY=0.025`
- Late layers (5-10): `MATRIX_LR_LATE=0.030`
- `BANK_SPLIT=5`

Early layers learn slower, stabilizing feature extraction. Late layers learn faster, refining task-specific representations.

### 2. BigramHash(2816 x 160) — wider projection

PR #1019 used BigramHash(3072 x 112). We use fewer buckets (2816) but a wider projection dimension (160 vs 112). The wider projection captures richer bigram representations while reducing artifact pressure from the hash table.

### 3. Sigmoid-Gated U-Net Skip Connections

Standard U-Net passes encoder outputs directly to decoder. We add a learnable sigmoid gate:
```python
g = sigmoid(gate)
output = g * x + (1 - g) * scaled_skip
```
Gates initialized to zero (starts as identity), learns to blend skip information.

### 4. Soft-Round QAT (alpha ramp 1->16)

Replaces standard STE quantization-aware training with a temperature-controlled soft-round:
```python
soft_round = floor(x) + sigmoid(alpha * (frac - 0.5))
```
Alpha ramps from 1 (smooth) to 16 (hard) over 500 steps, providing real gradients near quantization boundaries instead of the STE identity surrogate.

### 5. Brotli-11 + Byte-Shuffle Compression

Replaces LZMA preset=9 with Brotli quality=11 plus byte interleaving (stride=2). Saves ~400KB of artifact space vs LZMA, providing more headroom for model weights.

### 6. Code Minification (101KB -> 23KB)

Self-extracting LZMA-compressed wrapper reduces code from ~101KB to ~23KB, saving ~78KB of artifact budget for model weights.

### Negative Results

- **warmdown=4000**: +0.002 BPP worse than warmdown=3500 on this stack
- **BigramHash 3072×112** (PR #1019 config): +0.003 BPP worse with Brotli compression
- **LR floor 5%**: +0.001 BPP worse (prevents full convergence)
- **TTT (all configurations)**: Neutral or worse with Full Hessian GPTQ
- **No-GPTQ (Rascal approach)**: +0.005 BPP worse on our stack

---

## Architecture

| Component | Setting | First introduced by |
|-----------|---------|---------------------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 3× (1536) with LeakyReLU(0.5)² | [#493](https://github.com/openai/parameter-golf/pull/493) @parinzee |
| Attention | XSA on all 11 layers | [#478](https://github.com/openai/parameter-golf/pull/478) @gowtham0992 |
| BigramHash | **2816 × dim=160** | **This work** (concept: [#162](https://github.com/openai/parameter-golf/pull/162) @raahilshah) |
| Split-LR | **early=0.025, late=0.030, bank_split=5** | **This work** |
| Skip connections | **Sigmoid-gated U-Net** | **This work** |
| QAT | **Soft-round (alpha ramp 1->16)** | **This work** |
| RoPE | Partial (16/64 dims) | [#315](https://github.com/openai/parameter-golf/pull/315) @jfprincz |
| LN Scale | 1/sqrt(layer+1) | [#315](https://github.com/openai/parameter-golf/pull/315) @jfprincz |
| VE128 | Layers 9-10 | [#374](https://github.com/openai/parameter-golf/pull/374) @unnir |
| SmearGate | Position-mixing gate | [#65](https://github.com/openai/parameter-golf/pull/65) @aquariouseworkman |
| Weight avg | EMA(0.997) + SWA(every 50) | [#401](https://github.com/openai/parameter-golf/pull/401) @newjordan |
| Quantization | Full Hessian GPTQ int6 (training-data calibration) | Our PR #1019 (GPTQ: [#535](https://github.com/openai/parameter-golf/pull/535) @raahilshah) |
| Compression | **Brotli quality=11 + byte-shuffle** | **This work** (Brotli: [#1089](https://github.com/openai/parameter-golf/pull/1089) @mikeapedia) |
| Data Pipeline | Coprime-stride multi-shard loader | [#726](https://github.com/openai/parameter-golf/pull/726) @DeepReinforce |
| Code | **Minified self-extracting wrapper (23KB)** | **This work** |
| Warmdown | 3500 iterations | [#364](https://github.com/openai/parameter-golf/pull/364) @shikhar1729 |
| Optimizer | Parallel Muon + Parameter Banking | [#399](https://github.com/openai/parameter-golf/pull/399) @abaybektursun |
| Late QAT | STE at LR scale < 0.15 | [#286](https://github.com/openai/parameter-golf/pull/286) @chris-buckley |
| Flash Attention 3 | Hopper warp-specialized kernels | [#122](https://github.com/openai/parameter-golf/pull/122) @mtybadger |

## Rule Compliance

- ✅ Standard F.cross_entropy scoring (softmax, full distribution, sum=1)
- ✅ No TTT, no SLOT, no eval-time adaptation of any kind
- ✅ No validation data during training
- ✅ No training data during evaluation (GPTQ calibrates within 600s training budget)
- ✅ Single left-to-right sliding-window evaluation pass
- ✅ Artifact < 16,000,000 bytes (max: 15,811,712)
- ✅ Training ≤ 600s on 8×H100 SXM (591s)
- ✅ Eval ≤ 600s on 8×H100 SXM (~86s)
- ✅ 3-seed verification with low variance (std 0.0006)

## Run Command

```bash
# 3-seed reproduction
for SEED in 1337 42 2025; do
  BIGRAM_DIM=160 SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${SEED}.log
done
```

Environment: PyTorch 2.9.1+cu128, Flash Attention 3 (`flash_attn_interface`), NCCL_NET=Socket on GCP.

## Lineage

```
PR #549 (1.1194) — Parallel Muon + LeakyReLU² + legal TTT
    └── PR #1019 (1.1147) — AR self-gen GPTQ + XSA-all + BigramHash 3072×112
        └── This work (1.1105) adds:
            ├── Split-LR (early=0.025/late=0.030, bank_split=5)
            ├── BigramHash 2816×160 (wider projection, fewer buckets)
            ├── Sigmoid-gated U-Net skip connections
            ├── Soft-round QAT (alpha ramp 1→16)
            ├── Brotli-11 + byte-shuffle compression (+400KB headroom vs LZMA)
            ├── Code minification (101KB→23KB, saves 78KB artifact)
            └── GPTQ reserve reduced 14s→9s (+62 training steps)
```

## Credits

- **Base scaffold**: [PR #549](https://github.com/openai/parameter-golf/pull/549) and [PR #1019](https://github.com/openai/parameter-golf/pull/1019) by @abaybektursun
- **Coprime-stride loader**: [PR #726](https://github.com/openai/parameter-golf/pull/726) by @DeepReinforce
- **Full Hessian GPTQ**: [PR #535](https://github.com/openai/parameter-golf/pull/535) by @raahilshah
- **XSA-all**: [PR #478](https://github.com/openai/parameter-golf/pull/478) by @gowtham0992
- **LeakyReLU(0.5)²**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee
- **Brotli compression**: independently discovered; also in [PR #1089](https://github.com/openai/parameter-golf/pull/1089) by @mikeapedia

## Acknowledgements

Thanks to **@0hq** and **@valerio-oai** for organizing, maintaining, and moderating an unusually fun and technically demanding competition.

This submission benefited from reading and learning from other competitors' public work, especially the broader discussion around legal evaluation methods, quantization strategies, and the ongoing analysis in Issue #677 and Issue #1017.

## Included Files

- `train_gpt.py` — full training + quantization + evaluation script (23,349 bytes, minified self-extracting)
- `train_seed1337.log`, `train_seed42.log`, `train_seed2025.log` — all seed logs
- `submission.json` — leaderboard metadata
