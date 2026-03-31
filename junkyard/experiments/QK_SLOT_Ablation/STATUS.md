# QK_SLOT_Ablation — Status

**Date:** 2026-03-31  
**Branch:** TEST_LAB  
**Commit:** 86af1f3  
**Goal:** Validate two independent signals before spending $15 on 8×H100 race.

---

## What this experiment is

Single-GPU cross-correlation ablation with 4 cases:

| Case | QK_GAIN_INIT | SLOT | Signal |
|---|---|---|---|
| baseline | 1.5 (default) | off | control |
| qk_gain4 | 4.0 | off | training-side delta |
| slot_only | 1.5 | on | eval-side delta |
| qk_gain4_slot | 4.0 | on | cross-correlation |

**Expected deltas (from prior work):**
- QK_GAIN_INIT=4.0: ~-0.006 BPB (validated across 45 runs, 3 codebases)
- SLOT: ~-0.021 BPB (arXiv:2505.12392v2 — per-batch delta vector, score-first, legal)

**Cross-correlation check:** if `|(combo delta) - (qk delta + slot delta)| < 0.002` → compatible, both go in the race build.

---

## Files

- `train_gpt.py` — Rascal_Master base + SLOT spliced in (forward_hidden, compute_logits_from_hidden, modified eval_val_sliding)
- `run_ablation.py` — orchestrator: runs cases, parses logs, prints delta table + interaction residual, writes CSV
- `run_ablation.sh` — shell launcher with preflight checks

---

## Run config

- Seed: 444
- Steps: 1200 (no warmdown)
- SLOT_MAX_WINDOWS: 512 (~1M tokens, fast proxy)
- nproc: 1 (single GPU)
- ~30-60 min total on H100

---

## Current state

**RUNNING** on pod C.33906793 (Vast.ai H100).  
Logs land in `experiments/QK_SLOT_Ablation/logs/`.

---

## Base model

SOTA: Rascal II — 1.10986874 BPB, 15.44MB  
Script: `records/track_10min_16mb/2026-03-30_Rascal_8xH100/`  
PR: openai/parameter-golf#1120 — **LOCKED, do not touch**

---

## Next steps

1. Wait for ablation results
2. If both signals validate → build race script (Rascal_Master + QK_GAIN=4.0 baked + SLOT in eval)
3. One variable confirmed at a time before $15 race
