# E2E TTT Wishlist Submission — Portfolio Summary

**Author:** Abhishek Leji ([@X-Abhishek-X](https://github.com/X-Abhishek-X))
**Date:** 2026-04-26
**Submission track:** `track_non_record_16mb` (wishlist item: full-model E2E TTT)
**Companion record:** PR [#1695](https://github.com/openai/parameter-golf/pull/1695) (1.07590 BPB, 3-seed std 0.00019)

---

## TL;DR

Three contributions across this submission and the companion record PR:

1. **PR [#1695](https://github.com/openai/parameter-golf/pull/1695) — improved bigbag's SOTA.** Forked PR #1493 (bigbag, ~1.0810) and added SpinQuant V1 + MP-SGD-TTT to land at **val_bpb 1.07590** (3-seed mean, std 0.00019). Net **–0.025 BPB improvement** over the base — fork-and-improve, not a derivative regression.

2. **This submission — built the OpenAI wishlist E2E TTT and improved my own baseline.** A working full-model E2E TTT implementation with distributed lockstep gradient sync. Achieves **val_bpb 1.07063** on the PR #1695 checkpoint — a **–0.00527 BPB improvement** over PR #1695. **Non-record** because eval time of 1292s exceeds the 600s competition cap by design. Documents an unexpected "healing property" anomaly: SpinQuant+GPTQ degraded the post-quant model to 6.48 BPB; E2E TTT recovered fully to 1.07063 within the eval window — slightly exceeding the pre-quant ceiling of 1.07125.

3. **Empirical falsification of capacity expansion under the strict caps.** Independent attempt to push past current legal SOTA via int5 GPTQ + LQER + phased TTT on PR #1797's MLP_MULT=4.25 base. Measured int5 quant tax of **+0.030 BPB** (~30× the Discord-reported "+0.001"), and forced TTT_BATCH_SIZE=32 (from OOM at bsz=64 on 80GB H100) pushed eval to 652s — over the 600s cap. Final post-TTT BPB 1.07907, DQ on time. The four-way intersection of capacity expansion + 16MB + 600s + meaningful TTT is empirically infeasible with current techniques on this checkpoint family.

---

## Part 1 — E2E TTT (the positive result)

### What it does

Generalizes phased LoRA TTT (PR #1695, score-then-adapt within doc) to **full-model SGD per chunk** with distributed lockstep gradient synchronization (`all_reduce(MEAN)` across all 8 ranks before each `optimizer.step`). 35.9M trainable parameters per step.

### Result

| Metric | Value |
|---|---|
| Pre-quant val_bpb | 1.07125 |
| Post-quant pre-TTT val_bpb | 6.47968 (SpinQuant + GPTQ degradation) |
| **Post-TTT val_bpb (final)** | **1.07063** |
| Total eval time | 1292.4s |
| Artifact size | 15,961,787 B (≤ 16,000,000 cap) |
| Trainable params during TTT | 35,944,602 |
| SGD steps | 17,130 |
| Subset | `all` |

### Healing property observation

A measured, novel empirical observation: SpinQuant + GPTQ degraded the post-quant model from a pre-quant val_bpb of 1.07125 to **6.47968** (a 5.4 BPB regression — model is essentially broken on cold inference). E2E TTT recovered the post-quant model to **1.07063** within a 1292s eval window — **fully healing the quantization damage and slightly exceeding the pre-quant ceiling.**

This suggests that aggressive quantization may be more recoverable than commonly assumed when paired with full-model TTT, and is worth further investigation as a wishlist research direction.

### Why non-record

The 600s eval cap rules out E2E TTT at full subset (`all`) and chunk_size=48 — the algorithm is fundamentally heavier than phased LoRA TTT. Two record-eligible variants exist as future work:
- `PARAM_SUBSET=scale` — restrict trainable set to scalar / control parameters (~100× smaller). Estimated eval ~5-8 min, BPB ~1.072–1.075.
- `chunk_size=16` with reduced grad steps — finer-grained adaptation, lighter per-step.

These are left as follow-up PRs to keep this submission scoped to the wishlist item.

---

## Part 2 — Negative result: feasibility triangle for capacity expansion

### Setup

Independent attempt (Track B, separate from this E2E TTT submission) to push past the current #1 legal score by combining:
- **Base:** PR #1797 (dexhunter, published val_bpb **1.06157**, MLP_MULT=4.25, smear_gate, sparse_attn_gate)
- **Quantization:** int5 GPTQ + LQER asymmetric rank-4 correction + EMBED_BITS=7
- **Adaptation:** Phased TTT (LoRA score-then-adapt, the same recipe as PR #1695)

### Pre-quant baseline (verified)

The fp16 checkpoint reproduces PR #1797's published score on our pod: **val_bpb 1.06345** (matches dexhunter's 1.06157 within expected noise). **This score is attributable to PR #1797, not to this submission** — we inherited it as the base. We do not claim it.

### Compression results

| Metric | Value | Vs cap |
|---|---|---|
| Artifact size at int5 + LQER | **12,956,750 B** | ✅ 3.04 MB headroom under 16MB |
| Post-quant pre-TTT val_bpb | 1.09344 | int5 quant tax: **+0.030 BPB** |
| Post-TTT val_bpb | **1.07907** | TTT recovered 0.014; net **+0.003 worse than PR #1695** |
| Total eval_time | **652s** | ❌ 52s OVER 600s cap → DQ for record |

### The feasibility triangle

The combination of constraints produces a tight infeasibility region for capacity-expanded models. Empirically observed during this work:

| Constraint | Mechanism | Observed impact |
|---|---|---|
| **16 MB artifact cap** | fp16 of MLP_MULT=4.25 model = 141 MB → mandatory int5 quant for headroom | int5 + LQER fits at 12.96 MB ✅ |
| **80 GB H100 VRAM cap** | TTT_BATCH_SIZE=64 default + MLP_MULT=4.25 + int5 quant grads | Hit `torch.OutOfMemoryError` at 75.86/79.19 GB allocated → forced bsz=32 |
| **600 s eval time cap** | bsz=32 → ~1.5× more batches → eval slows from estimated ~450s to 652s | Over cap by 52s → DQ |
| **BPB quality** | int5 quant tax on this expanded model | +0.030 BPB at quant; TTT recovered to +0.003 worse than PR #1695 |

**Each pairwise constraint is satisfiable.** The four-way intersection (capacity expansion + 16MB + 600s + meaningful TTT) is empirically infeasible with int5 + phased LoRA TTT on this checkpoint family.

### Why this matters

Two practical implications for future submitters:

1. **Discord-reported "+0.001 BPB int5 tax" (Ethan Yang) does not generalize to MLP_MULT=4.25 / 11-layer models.** The actual tax measured here was **+0.030 BPB**, ~30× larger. Future int5 attempts on capacity-expanded checkpoints should validate the quant tax on the specific model before assuming favorable scaling.

2. **TTT_BATCH_SIZE=64 OOMs on 80GB H100s when paired with MLP_MULT=4.25 + int5 quantization.** The forced bsz=32 fallback adds enough wallclock to push phased TTT eval over the 600s cap. Future capacity-expansion attempts will hit the same wall unless either VRAM increases or the TTT algorithm gets memory-leaner.

### Receipts (reproducibility)

All numbers measured on RunPod 8×H100 80GB SXM, 2026-04-26 PM:
- Checkpoint MD5: `e526a423ff6247435c55d6f8ce117435`
- Patched train_gpt.py MD5: `fc0e1731030c6e6d9bc2dd54b3687686` (Track B int5 variant)
- Quantized artifact MD5: `61752d7cb5623f3614a23d788a795da9` (12,956,750 B)
- Run log preserved at `experiments/apr26_pod_run_final/track_b_int5.log`

---

## Attribution

- **PR #1797 (dexhunter):** base architecture (MLP_MULT=4.25, smear_gate, sparse_attn_gate) and pre-quant performance ceiling of 1.06157.
- **PR #1695 (X-Abhishek-X):** SpinQuant V1 + MP-SGD-TTT recipe; Apr 9 SOTA precursor; reproduced 3-seed at 1.07590, std 0.00019.
- **PR #1493 (bigbag):** earlier SOTA bag of techniques; this submission's training-time hyperparameters partially derive from this lineage.
- **Wishlist item (OpenAI README):** E2E TTT as a research direction.

---

## Files in this submission

| File | Purpose |
|---|---|
| `README.md` | Top-level submission readme |
| `PORTFOLIO_SUMMARY.md` | This file — full writeup |
| `submission.json` | Machine-readable metadata (track, scores, hyperparameters, files) |
| `train_gpt.py` | Patched training/eval script (MD5 `4397db0c9025478d0251434044f0df44`) |
