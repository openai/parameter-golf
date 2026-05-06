# v10 Moonshot Plan — 2026-03-26

## Goal
Beat merged SOTA (1.1228) and approach open-PR frontier (~1.07) using legal techniques only.
No prefill cache. All n-gram uses strict score-first/update-after.

## Files

| File | Description | Risk | Expected BPB |
|------|-------------|------|--------------|
| `train_gpt_v10_safe.py` | v9a + Hedge Mixer, no model scaling | Low | ~1.05–1.07 |
| `train_gpt_v10_moonshot.py` | 640-dim + ternary MLP + Hedge Mixer | Medium-High | ~0.95–1.05 |

## Techniques

### v10_safe (incremental over v9a)
- **Base**: v9a — 11-layer, dim=512, 11-gram backoff, entropy-adaptive alpha
- **Hedge Mixer**: multiplicative-weights online algorithm combining neural + n-gram
  - Weights (w_neural, w_ngram) adapt per-document via exp(-eta * loss_i)
  - eta=0.1, init=(0.5, 0.5), reset at boundary tokens
  - Expected gain: -0.005 to -0.010 bpb
- **Add-delta smoothing**: (count + 0.01) / (ctx_count + 0.01 * 1024) for n-gram
  - Avoids zero probabilities in low-count buckets
  - Expected gain: -0.001 to -0.003 bpb

### v10_moonshot (aggressive)
All v10_safe techniques PLUS:
- **model_dim=640** (was 512): ~42M params vs ~27M
  - num_heads=8, head_dim=80, num_kv_heads=4
  - mlp_mult=3.0 → hidden=1920
  - Expected gain over capacity alone: -0.03 to -0.05 bpb
- **Adaptive quantization** to fit 42M params in 16MB:
  - MLP weights: ternary {-1, 0, +1} stored as int8 (3 values → 5-7x zstd compression)
  - Attention Q/K/V/proj: Int4 [-7, 7] stored as int8 (15 values → 2-3x compression)
  - Embeddings/output/other: Int6 [-31, 31] (unchanged)
  - Estimated compressed size: ~11-14MB + ~80KB code = within 16MB

## Size Estimate (v10_moonshot)
- 42M params breakdown:
  - MLP (27M): ternary int8 raw=27MB → compressed ~5-7MB (only 3 distinct values)
  - Attn (13.5M): int4 int8 raw=13.5MB → compressed ~5-6MB (15 values)
  - Embed/other (1.5M): int6 → compressed ~0.7MB
  - Scales (FP16): ~300KB
  - Code: ~90KB
- **Total estimate: ~12-14MB** — risky but potentially fits

## Run Order

1. Run `train_gpt_v10_safe.py` first on 1xH100 to verify artifact size and basic correctness
2. Check: post-quant bpb roundtrip should be < 1.15
3. If OK, run on 8xH100 for final score
4. Run `train_gpt_v10_moonshot.py` on 8xH100 only if safe passes

## Fallback
If moonshot artifact > 16MB:
- Reduce model_dim to 576 (head_dim=72, ~34M params → ~10MB estimate)
- Or revert to dim=512 with just hedge mixer (v10_safe)

## Risks
- Ternary quantization may degrade quality significantly (UNVERIFIED on this arch)
- Int4 attention may hurt more than Int8 at 640-dim scale
- 640-dim may not converge as well within 10-min budget vs tuned 512-dim
- Always verify: pre-quant BPB vs post-quant BPB diff should be < 0.005

## Key Constraints Checklist
- [ ] Artifact < 16,000,000 bytes
- [ ] Training ≤ 600s on 8xH100
- [ ] Eval ≤ 600s on 8xH100
- [ ] No prefill cache
- [ ] N-gram: score-first, update-after
- [ ] N-gram: per-rank tables only (no cross-rank sharing)
- [ ] Val data: never accessed during training
