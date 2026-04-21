# Spec 017 — seed 42 execution notes

---

## ⚠️ RETROACTIVE FINDING (added 2026-04-21 after run completion)

**The post-TTT val_bpb of 1.06733 below was measured on a model where TTT forward path ignores recur_alpha.**

- `_block_with_lora` in `train_gpt.py` does not apply the α blend.
- `eval_val_ttt_phased` uses `forward_ttt` → `_block_with_lora` for both loss measurement and LoRA adaptation.
- Training learned α ≈ [[1.08, 1.27, 1.43], [1.02, 0.97, 0.83]], but TTT evaluated an effective α=1 model.
- Latent gap in spec 015's original recur-alpha patch (commit `a9aa141`), inherited here.

Not fixing 017's artifacts. Flagging so anyone reading the numbers later knows the pipeline was not fully consistent. A separate future spec will test whether fixing the gap improves or worsens post-TTT val_bpb.

---


**Pod:** 9crwq3fldt5tfj (AP-JP-1, 8×H100 SXM)
**Volume:** jlxvxeiol4 mounted at /workspace (not /runpod — default mount path used)
**Commit:** 4dd2d63

## Setup notes

- JP volume mounted at `/workspace` this session (not `/runpod` as in prior sessions). All spec paths substituted accordingly.
- Inductor cache was warm (1025 files from prior runs on this volume). First-step compile was fast.

## Training summary

- Final step: **4784** (vs 008 baseline 4828 — 44 steps short, ~0.9% throughput tax)
- Loop (recur_alpha) activated at step ~2200
- Pre-activation tok/s: ~8,070k — Post-activation: decayed from 7,951k to 6,331k by step 4700
- α final: [[1.078, 1.273, 1.430], [1.016, 0.973, 0.832]] — healthy spread, not collapsing
- α grad_norm: ~0.002-0.006 post-activation, consistent gradient flow

## Pipeline results

| Stage | val_bpb |
|---|---|
| Bare endpoint (step 4784) | 1.0691 |
| Post-EMA pre-quant | 1.06861 |
| Post-GPTQ | 1.07781 |
| **Post-phased-TTT** | **1.06733** |

GPTQ cost: +0.00920
TTT recovery: −0.01048 (from post-GPTQ)

## Decision bucket (per spec 017)

1.06733 falls in **(1.06710, 1.06910]** — "Inside gate but worse than #1736 (1.06610)"
Margin vs #1736: **+0.00123 worse**

Spec says: "Shelve submission path; keep mechanistic findings. Likely reflects TTT partial absorption."

## Anomalies

- No anomalies. Clean run end-to-end. TTT × recur-alpha composition worked (no OOM, no crash).
- Submission size: 15.98 MB (under 16 MB limit).
