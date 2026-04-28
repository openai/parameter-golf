# Experiment 0076_confidence_gated_blend

Parent: 0074_per_context_alpha_blend

## Question

Add a model-confidence gate on top of 0074's per-context α blend. When the model's max log_softmax for a position is ABOVE a threshold (model is confident), use model logits ALONE without blending. When below threshold (model is uncertain), use the per-context α blend.

Per the per-token analysis (`scratch/blend_probe/per_token_analysis.py` + `scratch/blend_probe/conf_gate_on_per_ctx.py`):
- Per-context α alone: BPB 1.9416 offline.
- + gate at threshold = -1.0 (12% of tokens gated): BPB 1.9378 offline (Δ -0.0038).
- + gate at threshold = -0.5 (8% gated): BPB 1.9384 (Δ -0.0032).

## Hypothesis [LIKELY]

Predicted single-seed val_bpb ≈ 0074 single-seed 1.9521 - 0.004 = ~1.948.

The gate's mechanism: where the model is already confident (~10-12% of tokens have model log2p > -1), the trigram blend slightly hurts (per per-token analysis). Skipping the blend on those tokens reclaims that small loss.

## Change

Single-file edit to `experiments/0076_confidence_gated_blend/train_gpt.py` (and/or modules/). New env vars:

- `CONF_GATE_THRESHOLD` (float, default -1e9 = no gating, byte-identical to parent 0074). When > -1e9, activates gating: blend only when model max_log_softmax < threshold.

In modules/trigram_side_memory.py `trigram_blend_loss_multi_K`:
- Compute `model_max_log2p = model_log2p.max(dim=-1).values` shape (B, L)
- `gate_mask = model_max_log2p < threshold` shape (B, L) bool
- For positions where gate_mask is True (model uncertain): use the existing blended log probs
- For positions where gate_mask is False (model confident): use model_log2p alone (no blend)
- Compute CE on the resulting per-position log probs

Set `CONF_GATE_THRESHOLD=-1.0` in env.sh (best from offline sweep).

## Disconfirming

- val_bpb_post_quant > 0074 single-seed 1.9521 → gate hurts.
- Crash on production-shape MPS — same trap as 0071.
- Smoke disagrees with offline reference.

If val ≤ 1.95 → mechanism works, promote.

## Notes from execution

**Implementation (2026-04-28):**

Changes:
- `modules/trigram_side_memory.py`: added `conf_gate_threshold` kwarg to both
  `trigram_blend_loss` (single-K, default -1e9) and
  `trigram_blend_loss_multi_K` (multi-K, default -1e9). When the threshold is
  > -1e8 (i.e. enabled), compute `model_max_log2p = log_softmax.max(-1) / LN2`,
  build `gate_mask = model_max_log2p < threshold` (True = uncertain → blend),
  and `torch.where(gate_mask, blended_log_at_target, model_log_at_target)`
  before mean-NLL. The gather/where runs on the per-position log-prob at the
  TRUE TARGET only — no full (B,L,V) reconstruction, so memory profile is
  unchanged from parent 0074.
- `train_gpt.py`: added `Hyperparameters.conf_gate_threshold` (env
  `CONF_GATE_THRESHOLD`, default -1e9 = no gating). Added
  `GPT._conf_gate_threshold` (default -1e9) and forwarded into both
  `trigram_blend_loss` and `trigram_blend_loss_multi_K` calls. Set in main()
  on `base_model._conf_gate_threshold` for both single-K and multi-K install
  paths. Added log line for `CONF_GATE_THRESHOLD`.
- `env.sh`: added `export CONF_GATE_THRESHOLD=-1.0` (best from offline
  sweep). All other 0074 settings preserved (PER_CONTEXT_ALPHA=1, ALPHA_TAU
  0.5, ALPHA_THRESH 3.0, ALPHA_MIN 0.30, ALPHA_MAX 0.85, multi-K K=3,4
  with weights 0.7/0.10/0.20).

Smoke (`_gate_smoke.py`, all four gates passed):
- GATE 1 production shape MPS (B=3, L=1024, full GPT triple-parallel
  recurrent): gate OFF loss 7.0355, gate ON loss 7.0355 (random init —
  loss values are not meaningful, just that forward succeeds + finite).
- GATE 1b MPS gate-OFF reproducibility: 0.00e+00 diff.
- GATE 2 byte-identity vs parent 0074: with `CONF_GATE_THRESHOLD=-1e9`
  (default) the loss matches the parent multi-K + per-ctx α path to 0.00e+00
  on the offline cached val. Parent BPB 1.9464 vs offline ref 1.9416 (Δ
  +0.0048, within ±0.005 — same small drift the 0074 smoke saw).
- GATE 3 gated BPB: with `CONF_GATE_THRESHOLD=-1.0`, gated BPB 1.9375
  vs offline ref 1.9378 (Δ -0.0003, well within ±0.005).
- Pack size: gate is inference-time only — no extra buffers vs 0074
  (5.85 MB raw / 2.44 MB brotli unchanged).

Deviations: none. The gate path implements the exact mechanism described
in the spec: `gate_mask = (model_max_log2p < gate_threshold)`,
`torch.where(gate_mask, blended_lp, model_lp)`. The smoke uses production
shape on MPS (B=3, L=1024) per the 0071 lesson and exercises gate ON +
gate OFF + byte-identity reproducibility.
