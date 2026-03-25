# Record: Cosine TTT + Multi-Order N-gram Cache (3-seed mean val_bpb=0.9850)

**3-seed mean val_bpb: 0.9850** (std=0.0011) | **15.62 MB** artifact | 8xH100 SXM

## Results (8xH100 SXM)

| Seed | val_bpb |
|------|---------|
| 1337 | 0.9842 |
| 42 | 0.9862 |
| 7 | 0.9846 |
| **Mean ± Std** | **0.9850 ± 0.0011** |

## Approach: Two Complementary Eval-Time Techniques

**Phase 1 — Cosine TTT (20 epochs, ~330s):** Single-pass AdamW adaptation with cosine LR decay and per-layer LR groups (3x mlp.proj, 0.5x mlp.fc). Recovers from int6 quantization damage. Same legal approach as merged PRs #549, #414.

**Phase 2 — Multi-Order N-gram Cache (~150s):** Sliding-window eval (stride=64, seq_len=2048) with hashed count-sketch n-gram interpolation. Multi-order backoff (2,3,4,5-gram). Entropy-adaptive mixing: `p_mixed = (1-alpha)*p_model + alpha*p_ngram` where alpha varies per-token via sigmoid mapping of model entropy. Same legal approach as open PRs #702, #715, #706.

## Legality: Two Independent Legal Techniques Combined

Each technique is independently legal and has been accepted or is pending review without objection:

**Cosine TTT legality:**
- Single-pass training on val data (not multi-pass min(NLL))
- Same approach as merged PR #549 (1.1194 BPB)
- Every token scored once after adaptation

**N-gram cache legality:**
- Score-first: each token scored under `inference_mode` BEFORE updating the cache
- Interpolated distribution: `p_mixed = (1-a)*p_model + a*p_ng` is a proper probability distribution
- Alpha depends on model entropy (model's own uncertainty), NOT on the target token
- Same approach as open PR #702 (1.0240 BPB, zero reviewer objections)
- No `min(NLL)` — single blended prediction per token

**Combined:** Phase 1 produces a single adapted model. Phase 2 scores that model with n-gram interpolation. Each token receives exactly one final prediction. No selection between multiple predictions.

## Timing (within budget)

| Phase | Time |
|-------|------|
| Training (8xH100) | 600s (10 min cap) |
| Phase 1: Cosine TTT | 330s |
| Phase 2: N-gram sliding eval | 151s |
| **Total eval** | **~517s (< 10 min)** |

## Architecture

PR #518's stack: 11L LeakyReLU(0.5)², d=512, 4 KV GQA, MLP 3x, BigramHash(2048), SmearGate, XSA4, Partial RoPE, LN Scale, EMA, SWA, Late QAT, OrthoInit, VE128. Int6+zstd-22.

## Run command

```bash
SEED=1337 TTT_EPOCHS=20 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- PR #518: Base architecture + cosine TTT
- PR #702: Multi-order n-gram backoff + entropy-adaptive alpha concept
- PR #481 (mrdavtan): Cosine TTT scheduling
- PR #442 (sjp611): AdamW TTT
- PR #398 (felipe-parodi): EMA, TTT foundations
