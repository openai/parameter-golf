# Record: MHA Path + 1855 9-hparam Stack + PR #1948 + PR #1855 (val_bpb = 1.06184, 3-seed)

> **Note:** This README captures only the bare submission record. The full
> set of insights from our parameter-golf run — every PR iteration we tried,
> the hyperparameter-tuning experiments behind each design choice, and the
> ablation results that drove our decisions — is being compiled into a
> detailed write-up. A more detailed write-up is at: https://www.junchengbillyli.com/llm-notes.html

**val_bpb (3-seed mean) = 1.06184** | σ ≈ 0.000379 | **~15.84 MB max (~15.84 MB mean)** | 8×H100 SXM | 600 s training + 600 s eval

A joint effort by **Tim Shen ([@TimS-ml](https://github.com/TimS-ml))** and **Billy Li ([@lijuncheng16](https://github.com/lijuncheng16))**, with thanks to **Prof. Lin Hao (Fordham University)** for sponsoring the **8×H100 SXM** and **4×RTX 4090** compute used in this submission, Xingyuan Ding for additional experiments, Bill (Yiyuan) Li for meaningful discussions on tokenizers, **Liju Yu ([@Lijun-Yu](https://github.com/Lijun-Yu))** for his invaluable insights, and **Hang Zhou ([@greyjoeyzhou](https://github.com/greyjoeyzhou))** for project discussions.

## TL;DR

Extends [PR #1948](https://github.com/openai/parameter-golf/pull/1948) (Tim Shen's & Billy Li *Leaky ReLU Slope + GPTQ Reverse-Cholesky Speedup*, val_bpb=1.06242) with [PR #1855](https://github.com/openai/parameter-golf/pull/1855)

Two algorithmically free wins:

1. **Leaky ReLU squared slope 0.5 → 0.3** — `−0.00073` BPB free win; size-neutral, wallclock-neutral. (4-point sweep confirms 0.3 is the minimum — see Key Change 1.)
2. **GPTQ reverse-Cholesky + triangular solve** instead of the standard `chol → cholesky_inverse → chol(upper)` — mathematically equivalent within fp32 ULP, **2.07–2.24× faster on RTX 4090 cuSOLVER microbench** at the GPTQ workload range. (Key Change 2.)

Both are hardcoded inside `train_gpt.py` (the variant from [PR #1867](https://github.com/openai/parameter-golf/pull/1867)), which also ships **this PR's compliance-tuned defaults on top of PR #1938**: `LQER_TOP_K=3`, `GATED_ATTN_QUANT_GATE=1`, `TTT_BATCH_SIZE=16`, `PHASED_TTT_NUM_PHASES=3`, `GPTQ_RESERVE_SECONDS=16`.

## Result

| Seed | **Post-TTT val_bpb (final)** | Artifact bytes | Eval time |
|------|-----------------------------:|---------------:|----------:|
| 1334 | **1.06222**                  | 15,844,523     | 591.0 s   |
| 999  | **1.06183**                  | 15,834,049     | 576.5 s   |
| 42   | **1.06146**                  | 15,843,016     | 588.2 s   |
| **3-seed mean** | **1.06184** (σ ≈ 0.000379) | **15,840,529** mean / 15,844,523 max | 585.2 s mean |


## GPTQ reserve-time accounting
> **(04-30):** We've noticed that several
> leaderboard submissions appear to exceed the 10-minute training cap once the
> full GPTQ pipeline (Hessian collection, quantization, serialize, compress) is
> accounted for. From our own measurements, `gptq_reserve_seconds=0.5s` is
> **far insufficient**: GPTQ Hessian collection takes **~3.5-4 s** (depending
> on calibration batch size), GPTQ quantization itself **~10 s**, and the
> serialize+compress step adds another **~60-70 s for Brotli** or **~90-100 s
> for lrzip pergroup**. Among the top leaderboard PRs we surveyed, observed
> `gptq_reserve_seconds` values range across **0.5 / 4 / 8 s**; this submission
> uses **16 s** so that the full pipeline completes inside the 600 s training
> cap with margin. The few-second discrepancy is unlikely to be large enough
> to materially change the leaderboard score or ranking, but we think it's
> worth flagging.

## Key Change 1 in PR1948: Leaky ReLU² slope = 0.3

4-point sweep at fixed seed=42 / 1.0× batch / 600 s wallclock:

| slope | TTT BPB | Δ vs 0.30 |
|------:|--------:|----------:|
| 0.25  | 1.06151 | +0.00012  |
| **0.30** | **1.06139** | 0     |
| 0.35  | 1.06192 | +0.00053  |
| 0.50 (prior baseline) | 1.06212 | +0.00073 |
| 0.70  | 1.06267 | +0.00128  |

Shallow V minimum at 0.3, size-neutral, no wallclock cost. Hardcoded in `train_gpt.py` lines 694-695 (Triton kernel) and line 910 (eager fallback).

## Key Change 2 in PR1948: GPTQ reverse-Cholesky Hinv path

Replaces

```python
Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))   # 1 chol + 2 tri-solve
Hinv = torch.linalg.cholesky(Hinv, upper=True)            # 1 chol on dense H^{-1}
```

with the mathematically equivalent single-pass

```python
H_flip = torch.flip(H, dims=(0, 1))
L_flip = torch.linalg.cholesky(H_flip)
U      = torch.flip(L_flip, dims=(0, 1))
Hinv   = torch.linalg.solve_triangular(U, eye, upper=True)
```

(The proof uses `chol(H^{-1}, upper)` uniqueness under the positive-diagonal constraint; full derivation in the authors' Stage 7 ablation note.)

**RTX 4090 cuSOLVER fp32 microbench:**

| n | baseline | reverse_cholesky | speedup |
|--:|---------:|-----------------:|--------:|
|  512 | 0.78 ms  | 0.38 ms | **2.07×** |
| 1024 | 1.80 ms  | 0.82 ms | **2.18×** |
| 2048 | 3.91 ms  | 1.75 ms | **2.23×** |
| 4096 | 12.99 ms | 5.81 ms | **2.24×** |

Numerics: max relative error ≤ 5.3e-7 across `n=64..2048`; artifact bytes byte-equivalent within brotli noise. Hardcoded in `train_gpt.py` lines 1870-1874.

## Compliance-tuned defaults (this PR vs PR #1948)

This PR's headline change is the **MHA conversion + 1855 9-hparam stack**: switch
from KV=4 GQA to KV=8 MHA, drop MLP_MULT 4.0→3.5 to stay cap-legal, then layer
on the 9-hparam tuning stack ported from PR #1855. 

### MHA path (new in this PR)

| Hparam | PR #1948 | This PR | Note |
|--------|---------:|--------:|------|
| `NUM_KV_HEADS` | 4 | **8** | MHA: KV heads = Q heads (8/8) |
| `MLP_MULT` | 4.0 | **3.5** | offsets +KV bytes; cap-legal at 8×H100 |

### 1855 9-hparam stack (new in this PR)

| Hparam | PR #1948 | This PR |
|--------|---------:|--------:|
| `WARMDOWN_FRAC` | 0.75 | **0.85** |
| `BETA2` | 0.95 | **0.99** |
| `TTT_BETA2` | 0.999 | **0.99** |
| `TTT_WEIGHT_DECAY` | 1.0 | **0.5** |
| `TTT_LORA_RANK` | 96 | **80** |
| `SPARSE_ATTN_GATE_SCALE` | 1.0 | **0.5** |
| `PHASED_TTT_PREFIX_DOCS` | 2000 | **2500** |
| `EMBED_CLIP_SIGMAS` | 15.0 | **14.0** |
| `MLP_CLIP_SIGMAS` | 12.0 | **11.5** |

### Existing (carried from PR #1948)

| Hparam | PR #1948 | This PR |
|--------|---------:|--------:|
| `LQER_TOP_K` | 1 | **3** |
| `GATED_ATTN_QUANT_GATE` | 1 | **1** |
| `TTT_BATCH_SIZE` | 16 | **16** |
| `PHASED_TTT_NUM_PHASES` | 3 | **3** |
| `GPTQ_RESERVE_SECONDS` | 16.0 | **16.0** |
| `CASEOPS_ENABLED` | 1 | **1** |
| `SPARSE_ATTN_GATE_ENABLED` | 1 | **1** |
| `COMPRESSOR` | `brotli` | **`pergroup`** (lrzip; ≈ −270 KB byte savings) |
| `EMBED_BITS` | 7 | **7** |
| `MIN_LR` | 0.1 | **0.1** |


## Architecture

SP8192 CaseOps + 11L **MHA(KV=8)/XSA11** + L3-5 depth recurrence x2 + L8+ parallel residual lanes + LeakyReLU(0.3)^2 MLP (mult=**3.5**) + ln-scale + tied embeddings + SmearGate BOS-safe + SparseAttnGate int8 (gate_scale=0.5) + GPTQ int6 Reverse-Cholesky/SDClip (mlp_clip=11.5) + embed int7 (clip=14) + LQER-asym rank4 top3 + **lrzip pergroup** + phased TTT LoRA r80 bs16 3ph (prefix_docs=2500, β₂=0.99, wd=0.5) + Adam β₂=0.99 + WARMDOWN_FRAC=0.85

Model size: **35,945,671** params (raw); ≈ **15.84 MB** compressed (lrzip pergroup).

## Quantization

Full-Hessian GPTQ + SDClip, on the reverse-Cholesky Hinv path:

- **GPTQ int6** (clip_sigmas=12.85): all attn (`c_q`, `c_k`, `c_v`, `proj`) and MLP (`fc`, `proj`) weights
- **GPTQ int7 + LQER asymmetric** (rank=4, factor int4, group_size=64): `tok_emb.weight` only (`LQER_TOP_K=3`)
- **Dedicated int8 row-quant**: `attn_gate_w` (`GATED_ATTN_QUANT_GATE=1`)
- **fp16 passthrough**: scalar params + small parameter weights
- **lrzip pergroup** final compression → artifact ≈ 15.84 MB (≈ −270 KB vs Brotli-11 baseline; validated by AB2 sweep 235604)

## Compliance (3-seed)

| Cap | Limit | Observed (max across 3 seeds) | Margin |
|-----|------:|------------------------------:|-------:|
| Artifact (decimal) | 16,000,000 bytes | 15,844,523 (s1334; s42=15,843,016; s999=15,834,049) | **155,477 bytes** |
| `train + GPTQ` | 600 s | 584.1 s training + 16 s GPTQ reserve = 600.1 s (all 3 seeds) | essentially at cap |
| `total_eval_time` | 600 s | 591.0 s (s1334) / 588.2 s (s42) / 576.5 s (s999) | **9.0 s (s1334) / 11.8 s (s42) / 23.5 s (s999)** |

> The MHA path with the 1855 hparam stack pushes against the eval cap (s1334 within 9 s) but stays compliant on all 3 seeds. The lrzip pergroup serializer recovers ≈ 270 KB of byte budget vs Brotli, which the MLP_MULT=3.5 + KV=8 conversion partially consumes. The 16 s GPTQ reserve is necessary to fit the full Hessian + quantize + lrzip-compress pipeline (see Note above).

## Dataset

This submission uses the **pre-built case-op augmented FineWeb-10B** tokenization from
[`romeerp/parameter-golf-caseops-v1`](https://huggingface.co/datasets/romeerp/parameter-golf-caseops-v1)
(pre-built shards), the same dataset that PR #1729 / PR #1736 / PR #1851 use.
The bijective case-op tokenizer (`fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`,
shipped in `tokenizers/`) and the build script (`prepare_caseops_data.py` +
`lossless_caps.py`) are included for byte-exact rebuild, but **using the
pre-built shards from `romeerp/parameter-golf-caseops-v1` is the recommended
path**.

## Reproducing

```bash
# Option A (recommended): use pre-built shards from HF.
huggingface-cli download romeerp/parameter-golf-caseops-v1 \
  --repo-type dataset \
  --local-dir ./data/datasets/fineweb10B_sp8192_caseops/

# Option B: rebuild locally with the shipped scripts: prepare_caseops_data.py

# Either way, the script expects shards at
# ./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
# (the path layout is preserved across both options).

export RUN_ID=repro_seed42
export SEED=42
torchrun --nproc_per_node=8 --standalone train_gpt.py
```

`Hyperparameters` defaults already encode this PR's compliance-tuned envelope (this PR + b-series, on top of PR #1938); no other env exports are needed.

## Builds On

| Layer | Origin |
|-------|--------|
| **PR #1948** ([@TimS-ml](https://github.com/TimS-ml) & [@lijuncheng16](https://github.com/lijuncheng16) - *Leaky ReLU Slope + GPTQ Reverse-Cholesky Speedup*, val_bpb=1.0624) | base submission stack |
| **PR #1938** ([@lijuncheng16](https://github.com/lijuncheng16) & [@TimS-ml](https://github.com/TimS-ml) — *S0/PR1851 + Cap Tokenizer + LQER + Global TTT*, val_bpb=1.0713 |
| **PR #1867** ([@lijuncheng16](https://github.com/lijuncheng16) & [@TimS-ml](https://github.com/TimS-ml)) | training script |
| **PR #1851** ([@aquariouseworkman](https://github.com/aquariouseworkman) — SmearGate BOS fix + LQER asymmetric + phased TTT) | architecture / quantization |
| PR #1797 ([@dexhunter](https://github.com/dexhunter), audit by [@cocohearts](https://github.com/cocohearts)) | SmearGate, LQER asym |
| PR #1787 ([@nprime06](https://github.com/nprime06)) | SparseAttnGate, FusedCE, MIN_LR |
| PR #1729 / PR #1736 ([@romeerp](https://github.com/romeerp)) | CaseOps tokenizer + phased TTT |
| PR #1394 ([@clarkkev](https://github.com/clarkkev)) | GPTQ + SDClip + SP8192 |
| PR #549 ([@abaybektursun](https://github.com/abaybektursun)) | Score-first TTT framework |

## Acknowledgments

A joint effort by **Tim Shen ([@TimS-ml](https://github.com/TimS-ml))** and **Billy Li ([@lijuncheng16](https://github.com/lijuncheng16))**.

With thanks to:

- **Prof. Lin Hao (Fordham University)** — for sponsoring the 8×H100 SXM and 4×RTX 4090 compute used to produce all sweep, training, and microbench results in this record.
- **Xingyuan Ding** — for experiments and A100 support.
- **Bill (Yiyuan) Li** — for meaningful discussions on tokenizers.
- **Liju Yu ([@Lijun-Yu](https://github.com/Lijun-Yu))** - for his invaluable insights.
- **Hang Zhou ([@greyjoeyzhou](https://github.com/greyjoeyzhou))** — for project discussions and for the concurrent auto-research agent infrastructure.

Additional credits (technique stack):

- [@aquariouseworkman](https://github.com/aquariouseworkman) — PR #1851 SmearGate BOS-fix base stack
- [@cocohearts](https://github.com/cocohearts) — SmearGate BOS audit (PR #1797)
- [@dexhunter](https://github.com/dexhunter) — SmearGate + LQER asymmetric, phased TTT (PR #1797 / PR #1736)
- [@romeerp](https://github.com/romeerp) — CaseOps tokenizer (PR #1729 / PR #1736)
- [@nprime06](https://github.com/nprime06) — SparseAttnGate / FusedCE / MIN_LR (PR #1787)
- [@abaybektursun](https://github.com/abaybektursun) — Score-first TTT framework (PR #549)
- [@clarkkev](https://github.com/clarkkev) — GPTQ + SDClip + SP8192 (PR #1394)
