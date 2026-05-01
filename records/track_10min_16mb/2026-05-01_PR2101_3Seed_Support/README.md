# 3-Seed Support Package — PR #2101

**val_bpb = 1.05879** (3-seed mean ± 0.00098 sample-std) | **15,983,343 B max** | 8×H100 SXM 80GB (RunPod)

This is a **3-seed independent reproduction and ablation report** for [PR #2101](https://github.com/openai/parameter-golf/pull/2101) by @OnlyJundong (PR #1855 base + AWQ-lite + AsymLogit + GradCentral + LabelSmooth code features). The PR's 3-seed claim is **1.05845 ± 0.00058**; this independent run lands at **1.05879 ± 0.00098** — within +0.00033 of the reported mean.

The package also contributes three **ablations on the unused author-introduced knobs** (`LABEL_SMOOTH`, `GRAD_CENTRALIZE`) plus a principled hparam probe (`TTT_LOCAL_LR_MULT`), all on seed 42. All three are null-or-negative — useful evidence that PR #2101's tuned stack is at a local optimum on these axes.

**No new ML technique is introduced.** This package reproduces PR #2101 exactly and ablates flags the author already added to the code.

## 3-Seed Reproduction (this run)

| Seed | PR #2101 claim | Our result | Δ |
|---|---:|---:|---:|
| 42   | 1.05897113 | **1.05792641** | −0.00104 |
| 0    | 1.05856489 | **1.05858896** | +0.00002 |
| 1234 | 1.05782711 | **1.05984791** | +0.00202 |
| **Mean** | **1.05845438** | **1.05878776** | **+0.00033** |
| **Std (sample, n−1)** | 0.00058 | **0.00098** | wider |
| **Std (population)** | — | 0.00080 | — |

Our 3-seed mean is **+0.00033 above** PR #2101's claimed mean — within typical pod-to-pod numerical drift. Our per-seed gap is roughly symmetric around zero (one beat, one match, one above), so the spread (not bias) is the dominant story.

## Compliance (vs project rules)

| Seed | Stop step | Train wallclock | Pre-quant BPB | Quant BPB | Total artifact bytes | TTT eval |
|---:|---:|---:|---:|---:|---:|---:|
| 42   | 4855 | 592.04 s | 1.06128455 | 1.06969200 | 15,979,586 | 408.6 s |
| 0    | 4857 | 592.08 s | 1.06208108 | 1.07042006 | 15,977,366 | 405.5 s |
| 1234 | 4856 | 592.14 s | 1.06296914 | 1.07165472 | 15,983,343 | 447.3 s |

- All `train_wallclock ≤ 600s` ✓ (592.04–592.14 s; 8.0 s reserved for GPTQ)
- All TTT eval `≤ 600s` ✓ (405.5–447.3 s)
- All artifacts `≤ 16,000,000 B` ✓ (worst-case 15,983,343 B; min slack 16,657 B)
- All 782 phased-TTT eval batches drained on every seed

## Comparison vs Currently-Merged SOTA (PR #1868, 1.06141)

| Seed | Our BPB | Δ vs 1.06141 |
|---:|---:|---:|
| 42   | 1.05792641 | **−0.00349** |
| 0    | 1.05858896 | **−0.00282** |
| 1234 | 1.05984791 | **−0.00156** |
| **Mean** | **1.05878776** | **−0.00262** |

Mean improvement vs merged SOTA: **−0.00262 BPB ≈ −0.00181 nats** — below the strict 0.005-nat record threshold, but **all 3 seeds individually beat merged SOTA** by 0.00156–0.00349 BPB.

## Ablations (seed 42)

PR #2101 added two new knobs to the codebase but disabled them in the 3-seed runs (default `GRAD_CENTRALIZE=0`, `LABEL_SMOOTH=0.0`). The README explicitly suggests `LABEL_SMOOTH=0.05` as an example value. We tested both, plus a principled `TTT_LOCAL_LR_MULT` probe in the direction PR #2060 was already moving.

| Variant | Seed 42 BPB | Δ vs baseline (1.05792641) | Verdict |
|---|---:|---:|---|
| **Baseline (GC=0, LS=0.0, TTT_LR_MULT=0.75)** | **1.05792641** | — | reference |
| `LABEL_SMOOTH=0.05` | 1.05800583 | +0.00008 | null (within noise) |
| `GRAD_CENTRALIZE=1` | 1.05796326 | +0.00004 | null (within noise) |
| `TTT_LOCAL_LR_MULT=0.85` | 1.05851093 | +0.00058 | worse |

Pre-quant deltas track the post-TTT deltas (label-smoothing slightly hurts pre-quant by +0.00022; grad-centralize +0.00009; TTT-LR doesn't affect pre-quant in principle, observed +0.00062 is run-to-run noise). Gradient centralization and label smoothing both leave training dynamics essentially unchanged on this stack — the post-TTT eval converges to the same number within ±0.00008. The TTT-LR=0.85 result tells us the 0.75 → 0.80 → 0.85 trajectory has flipped sign by 0.85; the local-LR optimum on this stack is between 0.75 and 0.80.

These three ablations (each one full 8×H100 run) cost ~$15 of compute and represent ~10% of project compute spent on negative results — useful evidence for the next contributor that these knobs are not load-bearing on PR #2101's tuned stack.

## Hardware / Environment

- 8× NVIDIA H100 80GB HBM3 SXM (RunPod)
- PyTorch 2.9.1+cu128
- CUDA 12.8
- FlashAttention 3 (cu128 build)
- lrzip per-group compression
- CaseOps SP8192 dataset on `/dev/shm`

## Reproduction

This package uses PR #2101's `train_gpt.py` unchanged (md5 `5606a60541ef66315ac6991e8cc16de8`). Hyperparameters match PR #2101 exactly:

```bash
SEED=42 \
MATRIX_LR=0.026 \
MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 \
ATTN_CLIP_SIGMAS=13.0 \
EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 \
LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_TOP_K=1 AWQ_LITE_GROUP_SIZE=64 \
ASYM_LOGIT_RESCALE=1 \
PHASED_TTT_NUM_PHASES=3 PHASED_TTT_PREFIX_DOCS=2500 \
TTT_MASK=no_qv TTT_LOCAL_LR_MULT=0.75 TTT_CHUNK_SIZE=48 \
TTT_LORA_RANK=80 TTT_LORA_ALPHA=144 \
GPTQ_RESERVE_SECONDS=8.0 GPTQ_CALIBRATION_BATCHES=16 \
MUON_BACKEND_STEPS=5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Replace `SEED=42` with `0` or `1234` for the other two seeds.

## What this submission is and is not

- ✅ Independent 3-seed reproduction of PR #2101 (mean within 0.00033 of claim)
- ✅ Confirmation that the technique beats currently-merged SOTA on every seed
- ✅ Three ablations of unused author-introduced knobs (negative results, useful for review)
- ❌ Not a record claim — mean is +0.00033 above PR #2101's claim and only −0.00181 nats below merged SOTA (below 0.005-nat threshold)
- ❌ No new ML technique introduced

## Credits

- **@OnlyJundong** — PR #2101 author (AWQ-lite + AsymLogit integration + GradCentral + LabelSmooth code features)
- **@codemath3000** — PR #1855 base
- **@aquariouseworkman** — PR #1851 (SmearGate BOS fix)
- **@dexhunter** — PR #1797 (SmearGate + LQER asymmetric)
- **@nprime06** — PR #1787 (base architecture)
- **@romeerp** — PR #1729 (CaseOps)
- **@clarkkev** — PR #1394 (GPTQ + SP8192)
- **@abaybektursun** — PR #549 (score-first TTT)
- **@cocohearts** — BOS document boundary bug identification

## Files

- `submission.json` — full per-seed and ablation results
- `train_seed42.log`, `train_seed0.log`, `train_seed1234.log` — 3-seed reproduction
- `ablation_label_smooth_005_seed42.log` — LABEL_SMOOTH=0.05 ablation
- `ablation_grad_centralize_seed42.log` — GRAD_CENTRALIZE=1 ablation
- `ablation_ttt_local_lr_085_seed42.log` — TTT_LOCAL_LR_MULT=0.85 ablation
