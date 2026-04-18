# Layer-wise LR decay

**Status:** candidate
**Expected Δ:** +0.002 to +0.005 (estimate; untested in this codebase)
**Source:** Community-proven in LLM pretraining literature; not clearly used in current SOTA submission.

## Idea
Apply different learning rates per layer depth: deeper layers get a smaller LR multiplier. Standard factor: `lr_layer_i = lr_base * decay^(num_layers - i)` with decay ∈ [0.75, 0.95].

## Why it might help
- Deeper layers tend to be closer to "task heads" and benefit from slower adaptation once early layers stabilize.
- Particularly helpful in small-budget regimes where optimizer must distribute limited step budget wisely.
- MuonEq-R (the SOTA optimizer) is per-parameter-group friendly; decay by group is natural to add.

## Code-change sketch
- In `train_gpt_sota.py`'s optimizer setup, group parameters by layer index.
- Apply `lr = base_lr * decay ** (n_layers - i)` per group.
- Add `llrd_decay` to Hyperparameters; disable with decay=1.0.

## Risks / open questions
- Interacts with warmdown schedule — LR decay is applied during warmup/warmdown too, or only to the base LR? Literature does it to the base; warmdown then scales all groups proportionally.
- Muon vs AdamW behavior may differ. Muon is already layer-aware (row normalization); LLRD may be less beneficial than with AdamW.
- Sweep decay ∈ {0.8, 0.9, 0.95} — three mini-test runs.

## If this works
Stacks cleanly with architectural changes.
