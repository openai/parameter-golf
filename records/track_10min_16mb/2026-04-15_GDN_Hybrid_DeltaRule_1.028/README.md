# Record: GDN-Hybrid (Gated DeltaNet + SWA) — val_bpb 1.0274 (2-seed mean)

**val_bpb: 1.0274** (2-seed mean, cold cache) | **~14.7 MB** | 8xH100 SXM, 590s | No TTT

Reproduction and independent verification of the GDN-Hybrid architecture (PR #1545).

## Results

### 2-seed cold-cache runs (fresh pods, verified Triton JIT overhead ~105s)

| Seed | Steps | EMA BPB | **Quantized BPB** | XSA BPB | Artifact (bytes) |
|------|-------|---------|-------------------|---------|-----------------|
| 1337 | 1857  | 1.018060 | **1.026927**     | 1.031282 | 15,524,240     |
| 42   | 1856  | 1.018499 | **1.027811**     | 1.033100 | 15,305,698     |
| **Mean** | — | **1.018280** | **1.027369** | **1.032191** | — |

Cold-start signature confirmed: step 1 at ~105-106s (Triton JIT overhead).
All artifacts under 16MB. Training 590s on 8xH100 SXM per seed.

### Supplemental warm-cache run (not part of submitted claim)

| Seed | Steps | EMA BPB | Quantized BPB | XSA BPB | Artifact (bytes) |
|------|-------|---------|---------------|---------|-----------------|
| 1337 | 2252  | 1.006084 | 1.014925     | 1.019328 | 15,994,883     |

Warm cache gives ~400 extra training steps due to no Triton JIT overhead.

## Architecture

**GDN-Hybrid (Model D):** `[GDN x5] -> [SWA] -> [GDN x5] -> [SWA_shared]`

- 12 layers total: 10 Gated DeltaNet + 2 Sliding Window Attention (weight-shared)
- Dimension: 512, MLP mult: 3x, GDN head_dim: 64
- SWA: window=512, 8 heads / 4 KV heads, weight-shared across both SWA layers
- QK-Gain: 5.0 (learnable per-head scaling)
- BigramHash(3072, 112) + trigram hash embeddings
- SmearGate on token embeddings
- Logit softcap: 30.0
- SP1024 tokenizer
- **Total parameters: 33,862,953**

The GDN layers maintain a recurrent key-value associative memory updated by the delta rule:
```
h_t = (I - beta_t * k_t * k_t^T) * h_{t-1} + beta_t * v_t * k_t^T
```

## Training

- **Optimizer:** Muon (Newton-Schulz 5) for matrices, AdamW for embeddings/scalars
- **Steps:** ~1857 in 590s (cold cache)
- **Batch:** 786,432 tokens (384 sequences x 2048)
- **EMA:** decay 0.997
- **VAL_LOSS_EVERY=9999:** no in-training validation evals

## Quantization

Full-Hessian GPTQ with int6 matrices + zstd-22 compression.
Quantization degradation: ~0.009 BPB.

## Compliance

Fixed predictor — no eval-time adaptation.

- Condition 1 (Causality): Sliding-window eval is strictly causal. GDN recurrent state is forward-only.
- Condition 2 (Normalized distribution): Standard softmax over full 1024-token vocabulary.
- Condition 3 (Score before update): N/A — no eval-time parameter updates.
- Condition 4 (Single pass): Each validation token scored exactly once.
- TTT_ENABLED=0, no SLOT, no RLS, no n-gram mixer
- GPTQ calibration uses model-generated synthetic sequences only
- All artifacts < 16,000,000 bytes

## Reproduction

```bash
pip install flash-linear-attention sentencepiece zstandard brotli
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
mkdir -p checkpoints

SEED=1337 ARCH_MODE=D MAX_WALLCLOCK_SECONDS=590 ITERATIONS=9999 \
  TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 \
  QK_GAIN_INIT=5.0 GPTQ_ENABLED=1 VAL_LOSS_EVERY=9999 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Novel Research: Error Correction Network (ECN)

Alongside this reproduction, we conducted extensive research into post-training logit correction methods. See [ECN_RESEARCH.md](ECN_RESEARCH.md) for full details.

Key result: a tiny online-learning correction network achieves **-0.039 BPB improvement** at zero artifact cost — larger than most TTT implementations in this challenge. The bottleneck is PyTorch autograd overhead on per-token updates; with a custom CUDA kernel this could run within the 10-minute eval budget.

Additional research includes adapters on random linear maps (OpenAI's README wishlist item) and systematic evaluation of 10+ post-training correction methods.

All research was conducted over **2 days** (April 13-14, 2026).

## Author

**Hamza Koyuer** ([@Hkoyuer](https://github.com/Hkoyuer)) — [Helolinks.com](https://helolinks.com)
HBO-ICT, Amsterdam University of Applied Sciences (HvA)

## Credits

Architecture and training code based on GDN-Hybrid by @dexhunter (PR #1545).
Independent reproduction and verification on fresh cold-cache pods.
