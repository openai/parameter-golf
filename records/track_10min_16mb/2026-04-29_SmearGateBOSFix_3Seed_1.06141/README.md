# Record: 3-Seed Compliance Reproduction — Support for PR #1851

**val_bpb = 1.06145** (3-seed mean ± 0.00068) | **~15.95 MB** | 8×H100 SXM 80GB

## Summary

This is a **3-seed compliance reproduction and support package** for [PR #1851](https://github.com/openai/parameter-golf/pull/1851) by @aquariouseworkman (SmearGate BOS Fix + PR #1787 Base + LQER Asymmetric + Phased TTT).

The purpose of this package is to:
1. Provide statistical significance evidence (3 seeds) for the PR #1851 result.
2. Confirm that results are reproducible across seeds by an independent party.
3. Document a compliance re-run demonstrating GPTQ fits within the 600s training budget.

**No new ML technique is introduced.** This package reproduces the exact code and configuration from PR #1851.

## 3-Seed Results (Original Runs)

These are the originally-committed results. Seed 42 is from @aquariouseworkman's PR #1851 submission; seeds 314 and 1234 were run by @Christopher-Lee-McClendon as independent reproductions using the same code and environment variables.

| Seed | Post-TTT BPB | Artifact (bytes) | Eval Time | Source |
|------|-------------|------------------|-----------|--------|
| 42   | **1.06128183** | 15,952,086 | 519.5s | PR #1851 (@aquariouseworkman) |
| 314  | **1.06086831** | 15,952,419 | 525.6s | Reproduction (@Christopher-Lee-McClendon) |
| 1234 | **1.06220261** | 15,952,690 | 479.6s | Reproduction (@Christopher-Lee-McClendon) |
| **Mean ± Std** | **1.06145 ± 0.00068** | | | |

All artifacts < 16,000,000 bytes ✓  
All eval times < 600s ✓

### Log Files (Original)

- `train_seed42_pr1851_original.log` — Seed 42 from PR #1851 by @aquariouseworkman
- `train_seed314_original.log` — Seed 314 reproduction by @Christopher-Lee-McClendon
- `train_seed1234_original.log` — Seed 1234 reproduction by @Christopher-Lee-McClendon

## Compliance Re-run Evidence (GPTQ Within 600s)

The original runs used `GPTQ_RESERVE_SECONDS=0.5`, which resulted in the training loop running until ~599.6s. GPTQ hessian collection (which accesses training data) adds ~3.5s, potentially extending past the 600s budget.

To confirm compliance, all 3 seeds were re-run with `GPTQ_RESERVE_SECONDS=8.0`, ensuring the training loop ends at ~592s and GPTQ hessians complete by ~595.5s (well within 600s). The only code change is the timing margin — no ML change.

| Seed | Post-TTT BPB (re-run) | Train Time | GPTQ Ends By | Artifact (bytes) |
|------|----------------------|------------|--------------|------------------|
| 42   | 1.06083288 | 592.1s | ~595.5s ✅ | 15,949,701 |
| 314  | 1.06090748 | 592.0s | ~595.5s ✅ | 15,951,777 |
| 1234 | 1.06248776 | 592.1s | ~595.5s ✅ | 15,951,968 |
| **Mean ± Std** | **1.06141 ± 0.00093** | | | |

**No statistically significant difference:** Original mean 1.06145 vs re-run mean 1.06141 (delta = −0.00004, well within 1-sigma noise). This confirms the GPTQ reserve setting has negligible impact on model quality.

### Re-run Log Files

- `train_seed42_rerun_gptq8s.log`
- `train_seed314_rerun_gptq8s.log`
- `train_seed1234_rerun_gptq8s.log`

### What Changed in Re-run

1. **`GPTQ_RESERVE_SECONDS` 0.5 → 8.0** — Training loop ends ~8s early for GPTQ headroom.
2. **Serialize-before-diagnostic reordering** — Artifact written immediately after GPTQ, before pre-quant diagnostic eval.
3. **Timing instrumentation** — `serialize_wallclock` and `artifact_production_wallclock` logged for transparency.

### GPTQ Timing Breakdown (Re-run)

| Phase | Time | Accesses Training Data? |
|-------|------|------------------------|
| Training loop (with 8s reserve) | ~592s | ✅ Yes |
| Hessian collection | ~3.5s | ✅ Yes |
| **Total training-data-access time** | **~595.5s** | **< 600s ✅** |
| Quantization | ~10.1s | ❌ No (uses cached Hessians) |
| Brotli compression | ~65-67s | ❌ No (pure I/O) |

## Technique Stack

All techniques inherited from PR #1851 and its lineage. No new techniques introduced.

| Technique | Source |
|-----------|--------|
| Base architecture (11L, MLP 4×, MuonEq-R) | PR #1787 (@nprime06) |
| SmearGate attention + BOS fix | PR #1797 (@dexhunter) + PR #1851 (@aquariouseworkman) |
| LQER Asymmetric quantization | PR #1797 (@dexhunter) |
| CaseOps SP8192 | PR #1729 (@romeerp) |
| GPTQ + SP8192 | PR #1394 (@clarkkev) |
| Score-first TTT (3 phases) | PR #549 (@abaybektursun) |
| BOS bug identification | @cocohearts |

## Architecture

11L × 512d × 8H/4KV, MLP 4×, LeakyReLU(0.5)², Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: layers 3–5 looped ×2 (activated at frac=0.35). Parallel residuals from layer 8. XSA on all 11 layers. SmearGate window=12.

## Reproduction

```bash
# Install dependencies
pip install brotli python-minifier

# Prepare CaseOps SP8192 data
python3 prepare_caseops_data.py  # downloads from romeerp/parameter-golf-caseops-v1

# Run training (replace SEED with 42, 314, or 1234)
SEED=42 \
CASEOPS_ENABLED=1 \
EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 \
EMBED_CLIP_SIGMAS=15.0 \
MLP_CLIP_SIGMAS=12.0 \
GPTQ_RESERVE_SECONDS=8.0 \
PHASED_TTT_NUM_PHASES=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Hardware:** 8×H100 SXM 80GB (RunPod)

## Credits

- **@aquariouseworkman** — PR #1851 author (SmearGate BOS fix, original seed 42 result)
- **@nprime06** — PR #1787 (base architecture)
- **@romeerp** — PR #1729 (CaseOps)
- **@dexhunter** — PR #1797 (SmearGate + LQER asymmetric quantization)
- **@cocohearts** — BOS document boundary bug identification
- **@abaybektursun** — PR #549 (score-first TTT)
- **@clarkkev** — PR #1394 (GPTQ + SP8192)
- **@Christopher-Lee-McClendon** — Seeds 314/1234 reproduction and compliance re-run
