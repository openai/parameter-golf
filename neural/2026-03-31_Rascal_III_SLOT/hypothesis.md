# Hypothesis: Rascal_III_SLOT
Date: 2026-03-31
Track: neural
Parent: neural/2026-03-30_Rascal_II/ (vault/train_gpt_rascal_sota_REAL.py)

## What changes (ONE variable)
SLOT_ENABLED: 0 → 1

Context-Only SLOT (legal variant) added to eval_val_sliding.

At each sliding window (except window 0):
1. Compute frozen hidden states from base model (no grad, weights unchanged)
2. Initialize delta = zeros(1, 1, dim), requires_grad=True
3. 8 steps AdamW: optimize delta via cross_entropy on context positions 0..wlen-stride-1 only
4. Score positions wlen-stride..wlen-1 under optimized delta.detach()

Window 0: base model only (no prior context to adapt from).
Training trajectory: identical to Rascal II. Only eval path changes.
Zero size cost. Zero training cost.

## Why
Gate result (QK_Gain_SLOT_Legal, 1-GPU, 1200 steps, seed=444, SLOT_MAX_WINDOWS=512):
  baseline:   1.38224 sliding_bpb
  slot_legal: 1.37655 sliding_bpb
  delta:      −0.00569

Real eval-side signal. Proxy inflation 5-15×. Full-run estimate: −0.0004 to −0.0011 BPB.
  At −0.0004: 1.10987 → 1.10947 — still beats #1089 (1.1091) comfortably
  At −0.0011: 1.10987 → 1.10877 — clear #1 territory

Legality: Context-Only SLOT is unambiguously score-first. Delta is optimized only on
already-scored positions. No tokens are peeked before scoring.

## Gate target (1-GPU, 2000 steps, seed=444)
- Paired A/B (baseline vs slot_legal, same pod, same seed): delta < −0.003
- Training loss curve: identical between arms (SLOT is eval-only)
- step_avg sanity: < 1820ms on 1×GPU (expect ~730ms × grad_accum)
