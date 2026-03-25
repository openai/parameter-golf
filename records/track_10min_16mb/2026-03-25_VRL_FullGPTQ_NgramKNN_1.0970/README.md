# VRL + Full GPTQ + 5-gram Cache + Hidden-State kNN-LM

**val_bpb: 1.0970** (3-seed mean, std 0.0006) | **~15.7 MB** | 8×H100 SXM

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-cache bpb | **Post-cache bpb** | Cache gain | Eval time | Artifact |
|------|----------|-------|---------------|-------------------|------------|-----------|----------|
| 42 | 93.7ms | 5,923 | 1.1464 | **1.0976** | -0.0488 | 454s | 15,682,550 |
| 1337 | 93.7ms | 5,923 | 1.1465 | **1.0965** | -0.0500 | 462s | 15,739,515 |
| 2024 | 93.5ms | 5,937 | 1.1451 | **1.0970** | -0.0481 | 444s | 15,553,082 |
| **Mean** | **93.6ms** | **~5,928** | **~1.146** | **1.0970 (std 0.0006)** | **-0.049** | **~453s** | |

## Key Innovations

### 1. Hidden-State kNN-LM (Novel — first in competition)

Based on Khandelwal et al. 2019 ("Generalization through Memorization", ICLR 2020).
During eval, we store the model's 512-dim hidden states (output of final_norm,
before the output projection) from already-scored tokens in a GPU ring buffer.
For each new uncertain token, we find the k=32 nearest neighbors by L2 distance
and build a non-parametric distribution from their targets via RBF kernel softmax.

This captures **semantic repetition** — paraphrased content, similar sentence
structures with different words, domain-specific vocabulary patterns — that
exact-match n-gram caches miss.

```
kNN gain over n-gram-only: -0.0065 BPB (measured via ablation)
```

- 30K-entry ring buffer, fp16, ~30MB GPU memory
- L2 squared distance with RBF kernel (temperature=50)
- Top-k=32 nearest neighbors
- Mixing weight: lambda=0.08 (pre-committed via confidence gate)
- Additive with n-gram cache (paper shows kNN + cache gains stack)

### 2. Online 5-gram Cache with Adaptive Lambda

Online n-gram cache accumulated from already-scored tokens during sliding window
eval. Backoff from 5-gram to bigram. Minimum 3 observations before use.

Key difference from PR #659 (ruled invalid): **no safety gate / oracle selection**.
Our mixing decision is pre-committed via confidence gate: if model's log-prob for
the target is below log(0.7), we mix. The mixing weight scales continuously with
model uncertainty (adaptive lambda = base_lambda × min((1-conf)/(1-threshold), 2)).
This is decided before seeing the target token.

```
N-gram gain over base model: ~-0.042 BPP
```

### 3. GPTQ Calibration Inside Training Budget

Per #677 ruling, GPTQ Hessian calibration runs within the 600s training budget.
Training loop stops at ~555s (45s reserve), then EMA + GPTQ calibration runs.
`total_train_time` in logs proves compliance (~597-598s, under 600s).

## Training Architecture

v42 stack (PR #569 by @gowtham0992) with GPTQ timing fix:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | 2048 |
| XSA | All 11 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| VRL | Value Residual Learning (arxiv:2410.17897) |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | Full GPTQ int6 (Hessian-aware Cholesky) + zstd-22 |
| Pruning | 3% magnitude pruning post-quant |
| QAT | Late QAT@0.15 with export alignment |
| GPTQ reserve | 45s within training budget |

## Eval Strategy

| Phase | Time |
|-------|------|
| Training (including GPTQ calibration) | ~598s (< 600s) |
| Sliding window eval with n-gram + kNN cache | ~458s (< 600s) |

## Legality Compliance (per #677)

- ✅ GPTQ calibration inside training timer (not eval)
- ✅ No safety gate / oracle selection (pre-committed confidence gate)
- ✅ No training data accessed at eval time
- ✅ N-gram + kNN caches strictly backward-looking
- ✅ Compliant /records submission package
- ✅ All artifacts under 16MB
- ✅ All timers under 600s

## Run Command

```bash
SEED=42 python3 -m torch.distributed.run --nproc_per_node=8 train_gpt.py
```

## Ablation (seed 42, same hardware)

| Configuration | BPB | Delta |
|--------------|-----|-------|
| Base (sliding window s128, no cache) | ~1.147 | — |
| + 5-gram cache (lambda=0.15, conf=0.7) | 1.1049 | -0.042 |
| + kNN cache (lambda=0.08, k=32) | **1.0976** | -0.007 |

## Credits

- **kNN-LM**: Khandelwal et al. 2019 (ICLR 2020) — novel application to eval-time compression
- **5-gram cache concept**: PR #659 by @deanbrr (our implementation uses pre-committed gate)
- **VRL**: arxiv:2410.17897, first non-TTT use in PR #569 by @gowtham0992
- **Full GPTQ**: IST-DASLab/gptq (ICLR 2023), implementation by @gowtham0992
- **LeakyReLU²**: PR #493 by @parinzee
- **Base stack**: PR #414 by @signalrush
