# Record: PR #1855 + AWQ-lite (PR #1918) + AsymLogit (PR #2060) + GradCentral + LabelSmooth

**val_bpb = 1.05845** (3-seed mean ± 0.00058) | **~15.98 MB** | 8×H100 SXM 80GB

## Summary

This PR integrates [AWQ-lite (PR #1918)](https://github.com/openai/parameter-golf/pull/1918) and [AsymLogit (PR #2060)](https://github.com/openai/parameter-golf/pull/2060) on top of the [PR #1855](https://github.com/openai/parameter-golf/pull/1855) base (@codemath3000, 1.06108 BPB), and additionally introduces:

- **Gradient centralization** — optional feature added to the Muon optimizer (`GRAD_CENTRALIZE`, default off).
- **Label smoothing** — optional training regularizer (`LABEL_SMOOTH`, default 0.0).
- **Longer eval time support** — configurable `PHASED_TTT_PREFIX_DOCS` and `PHASED_TTT_NUM_PHASES` for extended TTT eval within the 600s eval budget.

**3-seed improvement over PR #1855:** −0.00263 BPB (1.05845 vs 1.06108).

## 3-Seed Results

| Seed | val_bpb | val_loss | Artifact (bytes) | Steps | Eval Time |
|------|---------|----------|-----------------|-------|-----------|
| 42   | **1.05897113** | 2.31742136 | 15,978,610 | 4919 | 508.0s |
| 0    | **1.05856489** | 2.31709823 | 15,976,992 | 4920 | 520.0s |
| 1234 | **1.05782711** | 2.31712816 | 15,978,588 | 4917 | 510.2s |
| **Mean ± Std** | **1.05845 ± 0.00058** | 2.31722 ± 0.00018 | 15,978,063 | 4919 | |

All artifacts < 16,000,000 bytes ✓  
All eval times < 600s ✓

## Compliance (GPTQ Within 600s)

All seeds used `GPTQ_RESERVE_SECONDS=8.0`:

| Seed | Train stop | GPTQ hessians | Total data-access time |
|------|-----------|---------------|----------------------|
| 42   | 592.067s | +4.1s | **596.2s** ✅ |
| 0    | 592.067s | +4.1s | **596.2s** ✅ |
| 1234 | 592.067s | +4.1s | **596.2s** ✅ |

GPTQ quantization step (~10s) and Brotli/lrzip compression (~121s) do not access training data.

## Contributions in This PR

### AWQ-lite Integration (from PR #1918)

Scores each column group of a weight matrix by saliency = `act_rms × mean(|weight|)`, where `act_rms` is collected per-column during Hessian calibration. The top-K most salient groups (default K=1) are protected at int8 precision instead of the default int6/int7.

Applied to `tok_emb.weight`, which receives `gptq(int7) + awqgrpint8 + lqer_asym` treatment in this run. Technique originally from PR #1918.

Key env vars: `AWQ_LITE_ENABLED=1 AWQ_LITE_GROUP_TOP_K=1`

### AsymLogit Integration (from PR #2060)

Replaces the single `logit_softcap` scalar with two separate learnable parameters `softcap_pos` and `softcap_neg`, initialized from the original softcap value. Applied only on the eval/TTT forward path — training uses the original fused CE kernel unchanged for numerical stability. Both parameters stored as passthrough float16.

Technique originally from PR #2060.

Key env var: `ASYM_LOGIT_RESCALE=1`

### Gradient Centralization (new code feature, default off)

Subtracts the row mean from gradients inside the Muon optimizer before the Newton-Schulz step. Added as an optional feature; not enabled in these runs (`GRAD_CENTRALIZE=0`).

Key env var: `GRAD_CENTRALIZE=1` to enable

### Label Smoothing (new code feature, default off)

Blends cross-entropy targets with a uniform distribution during training. Handles both the fused Triton CE kernel path and the eager `F.cross_entropy` path. Not enabled in these runs (`LABEL_SMOOTH=0.0`).

Key env var: `LABEL_SMOOTH=0.05` (example)

## Technique Stack (Full)

All techniques from PR #1855 are inherited unchanged.

| Technique | Source |
|-----------|--------|
| Base architecture (11L, MLP 4×, MuonEq-R) | PR #1787 (@nprime06) |
| SmearGate attention + BOS fix | PR #1797 (@dexhunter) + PR #1851 (@aquariouseworkman) |
| LQER Asymmetric quantization (rank-4) | PR #1797 (@dexhunter) |
| CaseOps SP8192 | PR #1729 (@romeerp) |
| GPTQ int6 + int7 embed + per-group lrzip | PR #1394 (@clarkkev) |
| Score-first TTT (3 phases) | PR #549 (@abaybektursun) |
| 9-hparam greedy stack | PR #1855 (@codemath3000) |
| AWQ-lite | PR #1918 (integrated by @OnlyJundong) |
| AsymLogit | PR #2060 (integrated by @OnlyJundong) |
| **Gradient centralization (code feature)** | **This PR (@OnlyJundong)** |
| **Label smoothing (code feature)** | **This PR (@OnlyJundong)** |

## Architecture

11L × 512d × 8H/4KV, MLP 4×, LeakyReLU(0.5)², Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0 (asymmetric on eval path). Depth recurrence: layers 3–5 looped ×2 (activated at frac=0.35). Parallel residuals from layer 8. XSA on all 11 layers. SmearGate window=12. QK gain init=5.25.

## Reproduction

```bash
# Install dependencies
pip install brotli python-minifier

# Prepare CaseOps SP8192 data
python3 prepare_caseops_data.py

# Run training (replace 42 with 0 or 1234 for other seeds)
SEED=42 \
CASEOPS_ENABLED=1 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
PHASED_TTT_ENABLED=1 \
PHASED_TTT_PREFIX_DOCS=2500 \
PHASED_TTT_NUM_PHASES=3 \
EMBED_BITS=7 \
MATRIX_LR=0.026 \
MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 \
ATTN_CLIP_SIGMAS=13.0 \
EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 \
TTT_CHUNK_SIZE=48 \
WARMUP_STEPS=20 \
MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 \
WARMDOWN_FRAC=0.85 \
BETA2=0.99 \
TTT_BETA2=0.99 \
TTT_WEIGHT_DECAY=0.5 \
TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=8.0 \
GPTQ_CALIBRATION_BATCHES=16 \
VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 \
LQER_ASYM_ENABLED=1 \
LQER_RANK=4 \
LQER_FACTOR_BITS=4 \
LQER_ASYM_GROUP=64 \
LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 \
COMPRESSOR=pergroup \
NCCL_NET=Socket \
AWQ_LITE_ENABLED=1 \
ASYM_LOGIT_RESCALE=1 \
TTT_MASK=no_qv \
TTT_LOCAL_LR_MULT=0.75 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Hardware:** 8×H100 SXM 80GB (RunPod)

## Log Files

- `train_seed42.log` — Seed 42
- `train_seed0.log` — Seed 0
- `train_seed1234.log` — Seed 1234

## Credits

- **@aquariouseworkman** — PR #1918 (AWQ-lite)
- **@S0urC10ud** — PR #2060 (AsymLogit)
- **@codemath3000** — PR #1855 (base: 11L XSA + 9-hparam greedy stack)
- **@aquariouseworkman** — PR #1851 (SmearGate BOS fix)
- **@nprime06** — PR #1787 (base architecture)
- **@dexhunter** — PR #1797 (SmearGate + LQER asymmetric)
- **@romeerp** — PR #1729 (CaseOps)
- **@clarkkev** — PR #1394 (GPTQ + SP8192)
- **@abaybektursun** — PR #549 (score-first TTT)
- **@OnlyJundong** — This PR (integration, gradient centralization + label smoothing code features)
