# SP8192 CleanStack SGD-TTT

**Track:** 10min / 16MB  
**Target BPB:** ~1.07–1.08 (matching merged SOTA PR #1493 at 1.0810)  
**Branch:** `2026-04-16_SP8192_CleanStack_SGD_TTT`

---

## Architecture (11L × 512d × 8H/4KV)

| Component | Config |
|---|---|
| Layers | 11 (5 encoder + 6 decoder, U-Net skip connections) |
| Model dim | 512 |
| Heads | 8 query / 4 KV (GQA) |
| MLP | 4× (2048), LeakyReLU(0.5)² |
| Depth recurrence | L3–L5, ×2 passes, enabled at wallclock frac=0.35 |
| Parallel residuals | From L7 (GPT-J style) |
| XSA | All 11 layers |
| SmearGate | Yes |
| RoPE | Partial (16/64 dims) |
| LN scale | 1/√(layer+1) |
| Tied embeddings | Yes |
| Logit softcap | 30.0 |
| QK-Gain init | 5.25 |

## Training

| Hyperparameter | Value |
|---|---|
| Vocab | SP8192 (`kevclark/parameter-golf`) |
| Optimizer | MuonEq-R (NS5, row-norm) + AdamW for embeddings/scalars |
| matrix_lr | 0.022 |
| scalar_lr | 0.02 |
| muon_wd | 0.095 |
| embed_wd | 0.085 |
| muon_momentum | 0.99 (cosine warmup from 0.92 over 1500 steps) |
| EMA decay | 0.9965 |
| Warmdown | 72% of wall clock |
| Batch tokens | 786,432 |
| seq_len | 2048 |
| wall clock | 600s − 12s GPTQ reserve |

## Evaluation

| Stage | Method |
|---|---|
| Standard | Non-overlapping 2048-token windows |
| Sliding window | stride=64, full context |
| TTT | Score-first SGD chunk TTT: 3 epochs/32K-token chunk, lr=0.005, momentum=0.9, cosine decay, grad_clip=1.0 (Issue #1017 Track B compliant) |

## Quantisation

| Tensor | Format | Clipping | GPTQ |
|---|---|---|---|
| Attention / MLP matrices | int6 per-row | k=12.85σ (SDClip) | Full Hessian, block=128 |
| Token embeddings | int8 per-row | k=20.0σ (SDClip) | Non-Hessian |
| Control scalars | float16 passthrough | — | — |
| Compression | Brotli-11 (fallback LZMA-9) | | |

## Key improvements over Apr-8 submission (1.1156 BPB)

1. **SP8192 vocab** replacing SP1024 (~0.02 BPB gain from larger vocab coverage)
2. **MuonEq-R** row-normalised Muon (principled per-row scale invariance)
3. **MLP 4×** replacing 3× (more model capacity in same parameter budget)
4. **SGD chunk TTT** replacing LoRA AdamW TTT (simpler, faster, proven by SOTA)
5. **SDClip GPTQ** replacing percentile-search clipping (per-row σ-based clipping)
6. **int8 embeddings** replacing int6 (better embedding quality at ~same size)
7. **Brotli-11 compression** replacing LZMA-9 (better ratio, faster decode)
8. **Depth recurrence** with correct loop warmup (avoids compilation disruption)

## Reproduction

```bash
# 1. Install extras
pip install brotli sentencepiece

# 2. Download SP8192 data (one time)
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

# 3. Run leaderboard experiment (any seed)
SEED=42 bash records/track_10min_16mb/2026-04-16_SP8192_CleanStack_SGD_TTT/run_leaderboard_8xh100.sh

# 4. Smoke test (1 GPU, 60s)
bash records/track_10min_16mb/2026-04-16_SP8192_CleanStack_SGD_TTT/run_smoke_1gpu.sh
```

## TTT Compliance (Issue #1017)

The SGD TTT implementation satisfies all Track B conditions:
1. **Causality**: evaluation is strictly causal (standard cross-entropy on shifted token sequences)
2. **Normalised distribution**: standard softmax over full vocabulary, no score biasing
3. **Score-first**: each 32K-token chunk is fully scored under `torch.no_grad()` BEFORE any SGD update
4. **Single-pass**: each token is scored exactly once, in left-to-right order
5. **Cross-rank consistency**: gradients are all-reduced across all 8 ranks per SGD step
