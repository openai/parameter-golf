# Record: Leaky ReLU Slope + GPTQ Reverse-Cholesky Speedup + PR #1938 (val_bpb = 1.06242)

**val_bpb (3-seed mean) = 1.06242** | σ ≈ 0.00013 | **~15.95 MB** | 8×H100 SXM | 600 s training + 600 s eval

A joint effort by **Tim Shen ([@TimS-ml](https://github.com/TimS-ml))** and **Billy Li ([@lijuncheng16](https://github.com/lijuncheng16))**, with thanks to **Prof. Lin Hao (Fordham University)** for sponsoring the **8×H100 SXM** and **4×RTX 4090** compute used in this submission, and **Hang Zhou ([@greyjoeyzhou](https://github.com/greyjoeyzhou))** for project discussions.

## TL;DR

Extends [PR #1938](https://github.com/openai/parameter-golf/pull/1938) (Billy Li & Tim Shen's *S0/PR1851 + Cap Tokenizer + LQER + Global TTT*, val_bpb=1.0713) with two algorithmically free wins:

1. **Leaky ReLU squared slope 0.5 → 0.3** — `−0.00073` BPB free win; size-neutral, wallclock-neutral. (4-point sweep confirms 0.3 is the minimum — see Key Change 1.)
2. **GPTQ reverse-Cholesky + triangular solve** instead of the standard `chol → cholesky_inverse → chol(upper)` — mathematically equivalent within fp32 ULP, **2.07–2.24× faster on RTX 4090 cuSOLVER microbench** at the GPTQ workload range. (Key Change 2.)

Both are hardcoded inside `train_gpt.py` (the variant from [PR #1867](https://github.com/openai/parameter-golf/pull/1867)), which also ships **this PR's compliance-tuned defaults on top of PR #1938**: `LQER_TOP_K=1`, `GATED_ATTN_QUANT_GATE=1`, `TTT_BATCH_SIZE=16`, `PHASED_TTT_NUM_PHASES=3`, `GPTQ_RESERVE_SECONDS=16`.

## Result

| Seed | **Post-TTT val_bpb (final)** | Artifact bytes |
|------|-----------------------------:|---------------:|
| 1334 | **1.06257**                  | 15,947,664     |
| 42   | **1.06232**                  | 15,945,920     |
| 999  | **1.06237**                  | 15,946,532     |
| **Mean** | **1.06242** (σ ≈ 0.00013) | **15,946,705** |

## Key Change 1: Leaky ReLU² slope = 0.3

4-point sweep at fixed seed=42 / 1.0× batch / 600 s wallclock:

| slope | TTT BPB | Δ vs 0.30 |
|------:|--------:|----------:|
| 0.25  | 1.06151 | +0.00012  |
| **0.30** | **1.06139** | 0     |
| 0.35  | 1.06192 | +0.00053  |
| 0.50 (prior baseline) | 1.06212 | +0.00073 |
| 0.70  | 1.06267 | +0.00128  |

Shallow V minimum at 0.3, size-neutral, no wallclock cost. Hardcoded in `train_gpt.py` lines 694-695 (Triton kernel) and line 910 (eager fallback).

## Key Change 2: GPTQ reverse-Cholesky Hinv path

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

## Compliance-tuned defaults (this PR vs PR #1938)

| Hparam | PR #1938 | This | Reason |
|--------|---------:|-----:|--------|
| `LQER_TOP_K` | 3 | **1** | top-error matrix (`tok_emb`) only; −0.00044 BPB, saves bytes |
| `GATED_ATTN_QUANT_GATE` | 0 | **1** | int8 row-quant for `attn_gate_w`; −0.00011 BPB |
| `TTT_BATCH_SIZE` | 64 | **16** | smaller phased batch |
| `PHASED_TTT_NUM_PHASES` | 1 | **3** | −0.00118 BPB |
| `GPTQ_RESERVE_SECONDS` | 4 | **16** | observed Hessian (3.5 s) + quantize (12.2 s) ≈ 16 s; required for `train+GPTQ ≤ 600 s` |
| `LEAKY_RELU_SQ_SLOPE` (in script) | 0.5 | **0.3** | Key Change 1 |
| GPTQ Hinv path (in script) | `cholesky_inverse + chol(upper)` | **reverse Cholesky + tri-solve** | Key Change 2 |

All other hparams inherit from `train_gpt.py`'s `Hyperparameters` defaults, which match the PR #1938 envelope.

## Architecture

11L × 512d × 8H / 4KV, MLP 4× (2048 hidden), **LeakyReLU(0.3)²**. Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings (vocab 8192, caseops-augmented), logit softcap=30.0. Depth recurrence (loops layers 3-5, ×2, activated at frac=0.35). Parallel residuals from layer 8. Skip gates. SmearGate with BOS mask. Sparse attention gates. `model_params = 35,945,671`.

## Quantization

Full-Hessian GPTQ + SDClip, on the reverse-Cholesky Hinv path:

- **GPTQ int6** (clip_sigmas=12.85): all attn (`c_q`, `c_k`, `c_v`, `proj`) and MLP (`fc`, `proj`) weights
- **GPTQ int7 + LQER asymmetric** (rank=4, factor int4, group_size=64): `tok_emb.weight` only (`LQER_TOP_K=1`)
- **Dedicated int8 row-quant**: `attn_gate_w` (`GATED_ATTN_QUANT_GATE=1`)
- **fp16 passthrough**: scalar params + small parameter weights
- **Brotli-11** final compression → artifact ≈ 15.95 MB

## TTT

Phased TTT, 3 phases × 2000 prefix docs, score-first, Adam optimizer, cosine LR (peak 1e-4). LoRA rank=96 over K, MLP, O projections. `TTT_BATCH_SIZE=16`. The script's `total_eval_time` is the canonical eval timer (matches the convention used by past SOTA records).

## Compliance

| Cap | Limit | Observed | Margin |
|-----|------:|---------:|-------:|
| Artifact (decimal) | 16,000,000 bytes | 15,947,664 (max of 3 seeds) | 52,336 bytes |
| `train + GPTQ` | 600 s | 584.1 s + 15.6 s ≈ 599.7 s | ~0.3 s |
| `total_eval_time` | 600 s | 482.6 s / 485.6 s / 587.7 s | 12–118 s |

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
| **PR #1938** ([@lijuncheng16](https://github.com/lijuncheng16) & [@TimS-ml](https://github.com/TimS-ml) — *S0/PR1851 + Cap Tokenizer + LQER + Global TTT*, val_bpb=1.0713) | base submission stack |
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

- **Prof. Lin Hao (Fordham University)** — for sponsoring the **8×H100 SXM** and **4×RTX 4090** compute used to produce all sweep, training, and microbench results in this record.
- **Hang Zhou ([@greyjoeyzhou](https://github.com/greyjoeyzhou))** — for project discussions and for the concurrent auto-research agent infrastructure that drove the Stage 1–7 ablation sweeps in parallel.

Additional credits (technique stack):

- [@aquariouseworkman](https://github.com/aquariouseworkman) — PR #1851 SmearGate BOS-fix base stack
- [@cocohearts](https://github.com/cocohearts) — SmearGate BOS audit (PR #1797)
- [@dexhunter](https://github.com/dexhunter) — SmearGate + LQER asymmetric, phased TTT (PR #1797 / PR #1736)
- [@romeerp](https://github.com/romeerp) — CaseOps tokenizer (PR #1729 / PR #1736)
- [@nprime06](https://github.com/nprime06) — SparseAttnGate / FusedCE / MIN_LR (PR #1787)
- [@abaybektursun](https://github.com/abaybektursun) — Score-first TTT framework (PR #549)
- [@clarkkev](https://github.com/clarkkev) — GPTQ + SDClip + SP8192 (PR #1394)
