# WIP: Relaxed Recursive Transformer + Full SOTA Stack

**Status: In progress — compute grant pending. Results TBD.**

Target: sub-1.076 val_bpb | 8xH100 SXM | ≤16MB artifact

---

## Approach

This submission builds the full proven SOTA stack (SP8192, GPTQ SDClip, MuonEq-R, parallel residuals, QK-Gain, legal TTT) and adds one novel contribution: **Relaxed Recursive Transformers (RRT) with per-step LoRA adapters** on the depth-recurrence layers.

### Novel contribution: RRT-LoRA

Current depth recurrence submissions (PR #1394, #1437) loop layers 3-5 with identical shared weights. This works but has two problems:
1. Each recurrence pass is identical — the model can't specialize early vs. late passes
2. Quantization error amplifies across steps, limiting how many loops are stable

RRT-LoRA solves both. Each recurrence step applies a tiny learned LoRA delta (rank=4) to the shared attention Q and V projections:

```
Q_step_i = W_q(x) + lora_B_i(lora_A_i(x))
V_step_i = W_v(x) + lora_B_v_i(lora_A_v_i(x))
```

LoRA B matrices initialize to zero — identical to baseline at start of training — and learn to specialize each pass. Parameter cost: ~4K params per adapter pair, negligible against the 16MB budget.

Reference: Bae et al. "Relaxed Recursive Transformers" (ICLR 2025)

---

## Full Stack

Building on the current SOTA (PR #1394, #1412, #1413, #1445, #1493):

- **SP8192** tokenizer
- **11L x 512d x 8H/4KV**, MLP 4x, LeakyReLU(0.5)², tied embeddings
- **Partial RoPE** (16/64 head dims)
- **Depth recurrence** on layers 3,4,5 — 3 steps with RRT-LoRA adapters, activates at 35% of training
- **Parallel residuals** (GPT-J style) on layers 7+
- **QK-Gain 5.25** — learnable per-head query scaling
- **MuonEq-R** optimizer — row-normalized Muon (arXiv:2603.28254)
- **EMA decay 0.9965**
- **GPTQ SDClip** — int6 matrices (k=12.85), int8 embeddings (k=20.0)
- **Legal score-first TTT** — SGD (lr=0.005, momentum=0.9), 3 epochs per 32K-token chunk, cosine LR decay, first 2 layers frozen
- **LZMA code compression**
- **Tuned hyperparameters** — WD=0.095, MLR=0.022, warmdown=72%

---

## Architecture Summary

```
Physical layers:    11
Recurrent layers:   3, 4, 5 (3 steps each, LoRA rank=4)
Virtual layers:     ~17 (11 physical + 2×3 recurrent)
Parallel residuals: layers 7-10
LoRA params:        ~24K (negligible vs 16MB budget)
```

---

## Hypothesis

RRT-LoRA should improve on naive depth recurrence (PR #1493, 1.0810 bpb) by:
1. Allowing each pass to specialize — early pass coarse features, late pass fine-grained refinement
2. Reducing quantization error amplification — low-rank deltas are less sensitive to int6 than full matrices
3. Combining naturally with parallel residuals and TTT on the same base

Expected gain over current SOTA: 0.005–0.015 bpb.

---

## Why not Mixture of Depths?

MoD is compute-saving, not parameter-saving. The binding constraint is the 16MB artifact, not FLOPs. RRT-LoRA targets the same adaptive depth intuition but within the parameter budget constraint.

---

## Background

- Solo ML researcher / founder (VectaBind — AI drug discovery)
- Adaptive computation background: PonderNet/ACT-style architecture for multi-game reasoning
- Built SE(3)-equivariant EGNN + ESM2-3B on single NVIDIA L4
- GitHub: ChipGlitch

---

## Progress

- [x] Reviewed full leaderboard and SOTA stack (PRs #1394, #1412, #1413, #1437, #1445, #1493)
- [x] Designed RRT-LoRA architecture
- [x] Implemented full training script (train_gpt_rrt.py)
- [x] MuonEq-R optimizer
- [x] Legal score-first TTT
- [x] GPTQ SDClip quantization
- [x] EMA + LZMA compression
- [ ] H100 baseline validation (pending compute grant)
- [ ] RRT-LoRA ablation vs naive recurrence
- [ ] Hyperparameter sweep
- [ ] 3-seed statistical validation
- [ ] Final submission

---

## References

- Bae et al. (ICLR 2025). Relaxed Recursive Transformers
- MuonEq-R: arXiv:2603.28254
- clarkkev — SP8192 + GPTQ SDClip + MuonEq-R (PR #1394)
- dexhunter — 3-layer depth recurrence + legal TTT (PR #1437, #1413)
- Robby955 — Parallel residuals (PR #1412)
- X-Abhishek-X — Hyperparameter tuning (PR #1445)
- bigbag — 3-layer recurrence + parallel residuals + QK-Gain 5.25 (PR #1493)

---

## Planned Files

- `README.md` (this file)
- `submission.json`
- `train_gpt_rrt.py`
- `train_seed42.log`
- `train_seed314.log`
- `train_seed999.log`
