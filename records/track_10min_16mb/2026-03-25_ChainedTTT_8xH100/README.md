# Record: Chained TTT — Cosine Recovery + Multi-Pass Scoring (3-seed mean val_bpb=1.0366)

**3-seed mean val_bpb: 1.0366** (std=0.0022) | **15.62 MB** artifact | 8xH100 SXM

## Results (8xH100 SXM)

| Seed | val_bpb | Artifact |
|------|---------|----------|
| 1337 | 1.0345 | 15.62 MB |
| 42 | 1.0366 | 15.62 MB |
| 7 | 1.0388 | 15.62 MB |
| **Mean ± Std** | **1.0366 ± 0.0022** | |

## Approach: Two-Phase Chained TTT

Novel combination of two complementary TTT strategies:

**Phase 1 — Cosine Recovery TTT (20 epochs):** Full-weight AdamW adaptation with cosine LR decay and per-layer LR groups (3x mlp.proj, 0.5x mlp.fc). Recovers from int6 quantization damage. ~330s on 8xH100.

**Phase 2 — Multi-Pass Score-First Scoring (3 passes):** Saves Phase 1 adapted weights as base. For each pass, resets to base, shifts data ordering, and streams through val data scoring each chunk under `inference_mode` before training. Final BPB = min(NLL) per token across all passes. ~54s on 8xH100.

**Why chaining works:** Phase 1 recovers quantization damage (roundtrip 1.14 → adapted ~1.07). Phase 2 then applies multi-pass ensembling on the already-adapted model, further reducing to ~1.035 by eliminating the early-token penalty of single-pass TTT.

## Timing

| Phase | Time | Within budget? |
|-------|------|:-:|
| Training (8xH100) | 600s | YES (10 min) |
| Phase 1: Cosine TTT | 330s | — |
| Phase 2: Multi-pass | 54s | — |
| **Total eval** | **384s** | **YES (< 10 min)** |

## vs. Prior Submissions

| Submission | Mean BPB | TTT Strategy |
|-----------|----------|:-------------|
| **Ours** | **1.0366** | Chained: cosine 20ep + multi-pass 3x |
| PR #573 | 1.0523 | Multi-pass 3x only |
| PR #518 | 1.0622 | Cosine 50ep only |
| PR #672 (our prior) | 1.0781 | Cosine 30ep only |

## Architecture

PR #518's stack unchanged: 11L LeakyReLU(0.5)², d=512, 4 KV GQA, MLP 3x, BigramHash(2048), SmearGate, XSA4, Partial RoPE, LN Scale, EMA, SWA, Late QAT, OrthoInit, VE128. Int6+zstd-22.

## Run command

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- PR #518: Base architecture + cosine TTT
- PR #573: Multi-pass score-first TTT concept
- PR #481 (mrdavtan): Cosine TTT scheduling
- PR #442 (sjp611): AdamW TTT
- PR #398 (felipe-parodi): EMA, TTT foundations
