# Hypothesis: RASCAL_WINDOWN_TESTING
Date: 2026-03-31
Track: neural
Parent: neural/2026-03-30_Rascal_II

## What we are testing (Legal Window Strategy Suite)

This is a multi-arm eval-time strategy gate, NOT a single-variable hypothesis.
Each arm changes ONE thing vs the CTRL baseline during sliding window evaluation.
Training is identical across all arms (same seed, same data, same 120s budget).

| Arm | Strategy | Variable |
|-----|----------|----------|
| CTRL-00 | No adaptation | baseline |
| SLOT-01 | Legal context-only SLOT | SLOT_ENABLED=1 |
| SCALE-02 | Score-first Scale TTT | SCALE_TTT_ENABLED=1 |
| SLOT+SCALE-03 | Both combined | SLOT_ENABLED=1 + SCALE_TTT_ENABLED=1 |

## Legal SLOT (arm 01)
Context-only delta: optimize 1×1×dim additive bias on context positions (0..wlen-stride-1),
score only new positions (wlen-stride..wlen-1). Skip window 0 (no prior context).
Proven causal: hidden[t] depends only on tokens[0..t].
Prior proxy signal: −0.0057 BPB at 1200 steps (QK_Gain_SLOT_Legal gate).

## Scale TTT (arm 02) — first test
Rascal's RMSNorm has no learnable params. The analog is the Adam-trained scale params:
  attn_scale (dim=512, one per block × 11 blocks = 5632 params)
  mlp_scale  (dim=512, same = 5632 params)
These are NOT Muon-trained, so AdamW TTT is on-manifold (no manifold mismatch).

Mechanism: per-chunk, score-first.
  1. Score all windows in chunk with current attn_scale/mlp_scale.
  2. Train only those params on chunk tokens (lr=1e-4, 1 epoch, AdamW).
  3. Carry updated params to next chunk.

Why this might work where full-weight TTT failed:
  - No Muon manifold mismatch (attn_scale/mlp_scale are Adam-trained)
  - Minimal forgetting risk (scale params control OUTPUT MAGNITUDE, not representation)
  - 11264 params vs millions for full-weight TTT
  - Each chunk calibrates the model's dynamic range to the current distribution

## Gate target
Primary: SCALE-02 < CTRL-00 on final_sliding_window_exact val_bpb
Signal threshold: > 0.0005 BPB improvement (above proxy noise floor)
Bonus: SLOT+SCALE-03 < SLOT-01 (Scale TTT adds on top of SLOT)

## Notes
- This gate uses 2-min proxy runs (WALLCLOCK=120). Signals inflate ~5-10× vs full run.
- If SCALE-02 shows any positive delta, that is significant.
- SLOT-01 serves as a sanity check — we expect it to match prior −0.0057 proxy.
- "non Ngram": Ngram/bigram features are unchanged. This suite tests ONLY window strategy.
