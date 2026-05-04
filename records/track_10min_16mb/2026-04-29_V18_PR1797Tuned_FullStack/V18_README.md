# V18: PR #1797 BOS-fixed + Tuned Hparams (PR #1586/#1787/#1886)

**Strategy**: Fork dexhunter PR #1797 (BOS-fixed, 1.06412) unchanged code + tune hparams from 4 other clean PRs.

## Stack components (all CONFIRMED LEGAL)

| Component | Source | Value |
|-----------|--------|-------|
| Base architecture | PR #1797 dexhunter | unchanged |
| CaseOps tokenizer | PR #1797 / #1729 | bundled |
| Polar Express NS | PR #1787 nprime06 | inherited |
| MIN_LR=0.10 | PR #1787 | TUNED |
| Fused CE Triton | PR #1787 | inherited |
| Sparse Attn Gate | PR #1787 | inherited |
| SmearGate + BOS fix | PR #1797 / #1855 | inherited |
| LQER Asym int4 | PR #1797 | inherited |
| Phased TTT warm-start A | PR #1767 / #1797 | inherited |
| Per-Layer Adaptive GPTQ | PR #1586 dexhunter | TUNED |
| TTT WD=2.0 fix | PR #1886 renqianluo | TUNED |

## Hparam changes vs PR #1797 defaults

| Param | PR #1797 default | V18 value | Source | Reason |
|-------|------------------|-----------|--------|--------|
| MIN_LR | 0.0 | **0.10** | PR #1787 | Warmdown floor |
| MLP_CLIP_SIGMAS | 10.0 | **12.0** | PR #1586 | Tighter MLP clip |
| EMBED_BITS | 8 | **7** | PR #1586 | Save ~530KB |
| EMBED_CLIP_SIGMAS | 20.0 | **15.0** | PR #1586 | Pair with int7 |
| GPTQ_RESERVE_SECONDS | 4.0 | **0.5** | PR #1787 | More train time |
| TTT_WEIGHT_DECAY | 1.0 | **2.0** | PR #1886 | Prevent collapse with fused CE |

## Compliance (Issue #1017 Track A)

- [x] **Causality**: VarLen + per-doc cu_seqlens
- [x] **Normalized softmax**: full vocab
- [x] **Score-before-update**: TTT scored under no_grad before LoRA step
- [x] **Single pass**: each token scored exactly once
- [x] **No SLOT, no pre-quant TTT, no n-gram cache**
- [x] **Issue #1604** (CaseOps): inherited from PR #1797 (cocohearts audited PR #1797 only requested BOS fix)

## Expected Result

| Metric | dexhunter PR #1797 | V18 Estimate |
|--------|-------------------:|-------------:|
| Sliding val_bpb | 1.06412 | ~1.057-1.062 |
| Improvement vs PR #1797 | — | -0.002 to -0.007 |
| vs merged SOTA (1.0810) | -0.017 | ~-0.020 to -0.024 |
| Record threshold ✓ | -0.012 below | -0.015 to -0.019 below |

## Reproduction

```bash
cd records/track_10min_16mb/2026-04-29_V18_PR1797Tuned_FullStack/
bash run_v18_scout.sh        # single seed (42), ~12 min train + 5 min eval
bash run_v18_3seeds.sh       # full 3-seed validation, ~50 min total
```

## Attribution

- @dexhunter (PR #1797 base + PR #1586 GPTQ tuning + LQER Asym + SmearGate)
- @nprime06 (PR #1787 — Polar Express NS, MIN_LR, Fused CE, Sparse Attn Gate)
- @renqianluo (PR #1886 — WD=2.0 fix for fused CE + warm-start stability)
- @MarioPaerle (PR #1667 — Attention Output Gate concept; not used due to mutex with sparse_attn_gate)
- @samacqua (PR #1530 — VarLen + Triple Recurrence)
- @bigbag (PR #1493 — merged SOTA)
- @clarkkev (PR #1394 — SP8192 + GPTQ + SDClip)

This PR is a hyperparameter optimization of PR #1797's stack, combining tuning insights from 3 independent clean PRs (#1586, #1787, #1886) without any architectural changes.
