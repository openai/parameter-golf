# E2E TTT — `KS_E2E_TTT`

OpenAI Requests-for-PRs item: *"State-space models, E2E TTT, super
long context for evaluation or training."*

## What this is

The PR #1855 / #1953 phased TTT eval is **two-tier**:

1. *Per-doc TTT*: a small per-doc LoRA (rank 80, `B` learnable)
   adapts to each document during the score-first window.
2. *Per-phase global SGD*: between phases, a global SGD step trains the
   FULL base model on already-scored prefix docs.

So the recipe **already** does end-to-end (full-parameter) TTT — just
sandwiched between per-doc LoRA passes. `KS_E2E_TTT=1` would *also*
make the per-doc TTT inner loop full-parameter (rather than LoRA-only).

## Toy vs real

- **Toy hook (this submission):** the env var is read into the hparams
  but the existing TTT loop in `eval_val_ttt_phased` builds a
  `BatchedTTTLoRA` regardless. Wiring `KS_E2E_TTT=1` to swap the
  optimizer's parameter list to `base_model.parameters()` is the
  follow-up — surgical change to ~5 lines in `eval_val_ttt_phased`.
- **Real:** full E2E per-doc TTT was tried in earlier PRs (#303, "Record
  2" in the user's CLAUDE.md notes) and consistently *underperformed*
  LoRA-only TTT — full-weight per-doc updates destroy the SWA / EMA
  smoothing the base model accumulated, and there's no way to undo
  them between docs without saving the full base.

## Why it's still here

The Requests-for-PRs entry pairs E2E TTT with SSMs, suggesting OpenAI
wants to see *more* full-parameter test-time learning, not less. With
SSMs (which lack the heavy compositional structure attention has) the
"full-weight TTT destroys the base" failure mode might not bite as
hard. A real E2E TTT submission probably wants to be paired with a
state-space architecture and a smaller LR — that's the future PR.

## Limits

The implementation as currently wired (toggle read into hparams, no
optimizer swap yet) is the smallest honest scaffold. Anyone iterating
on this would need to:

1. Branch the TTT optimizer construction in `eval_val_ttt_phased`.
2. Snapshot base-model state at the start of each phase / batch.
3. Restore the snapshot after the per-doc adaptation, *or* let
   adaptation drift and verify it doesn't hurt later docs.
