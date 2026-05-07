# Stage 3 + SpinQuant V1 + MP-SGD-TTT

## Score: mean val_bpb = 1.07590 (3 seeds: 1.07591, 1.07609, 1.07570)

Trained on 8×H100 80GB SXM in 587 seconds. Artifact ~15.73 MB (INT6 + brotli).

## Approach

Two techniques stacked on the Stage 3 depth-recurrence base (PR #1445):

### 1. SpinQuant V1 — Hadamard Pre-Rotation Before GPTQ

Pre-multiplies Q, K, V weight matrices with a random Hadamard matrix `R` before INT6 GPTQ quantization, spreading weight outliers uniformly across all dimensions. This reduces the quantization error for the most outlier-heavy attention projections.

- `R` is generated deterministically from a SHA-256-derived seed and stored as `persistent=False` buffer — **zero serialized bytes added to the artifact**
- At eval time, `F.linear(x @ R, W_rot)` is equivalent to `F.linear(x, W)` (verified: max relative error < 1e-4)
- Hessian transform: `H_rot = R^T H R` applied before GPTQ for correct calibration in the rotated frame
- Quantization penalty: +0.012–0.013 BPB vs pre-quant baseline (suppressed by MP-SGD-TTT)

### 2. MP-SGD-TTT — Multi-Phase Global SGD Test-Time Training

Score-first causal TTT from PR #1626 (dexhunter). Three SGD phases over the validation stream:
- Each phase processes the already-scored prefix of documents
- Base model weights updated (not just LoRA) via momentum SGD
- Config: `prefix_docs=2000`, `num_phases=3`, `lr=0.001`, `momentum=0.9`
- BPB accumulated under `torch.no_grad()` before any gradient update on each chunk

## Results

| Seed | Pre-quant BPB | Post-quant BPB | TTT BPB | Artifact Size |
|------|:---:|:---:|:---:|:---:|
| 42   | 1.07288 | 1.08544 | **1.07591** | 15,728,308 B |
| 1337 | 1.07306 | 1.08584 | **1.07609** | 15,726,192 B |
| 2024 | 1.07273 | 1.08521 | **1.07570** | 15,727,886 B |
| **Mean** | | | **1.07590** | 15,727,462 B |
| **Std** | | | **0.00019** | |

All artifacts well under 16,000,000 bytes (decimal).

## Training Config

```
ITERATIONS=20000, MATRIX_LR=0.026, WARMDOWN_FRAC=0.75
MLP_CLIP_SIGMAS=12.0, ATTN_CLIP_SIGMAS=13.0, EMBED_CLIP_SIGMAS=20.0
EMBED_BITS=7, SPINQUANT_ENABLED=1, SPINQUANT_SEED=20260416
TTT_CHUNK_SIZE=48, TTT_LORA_LAYER_LR_ALPHA=0.5, LORA_PLUS_RATIO=1.0
```

## Attribution

- Stage 3 architecture: PR #1445 (X-Abhishek-X)
- MP-SGD-TTT: PR #1626 (dexhunter)
- SP8192 tokenizer: PR #78 (mtybadger)
- SpinQuant: Liu et al., Meta AI 2024 (arXiv:2405.16406)
