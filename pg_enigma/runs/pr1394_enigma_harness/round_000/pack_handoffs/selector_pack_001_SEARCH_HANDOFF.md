# Search Harness Handoff for selector_pack_001

- **Round**: `0`
- **Pack**: `selector_pack_001`
- **Pack kind**: `selector_pack`
- **Lane**: `selector`
- **Phase window**: `post_train`

```text
Use pg_enigma round 0 pack selector_pack_001 as the execution contract.
Pack kind: selector_pack
Lane: selector
Phase window: post_train

Create exactly these control slots and keep them unpatched:
- C0
- C1

Create candidate slots using these exact slot ids:
- H00: trajectory_003/trajectory_003_h3 / trajectory_003_h3_r1 (Late-Window Crossed Checkpoint x Export Sweep)
  family_group: checkpoint_export_selector
  rationale: Run immediately after control-delta calibration because it is executable now, lane-aligned, and tests a first-order searchable object without inventing new training machinery.
  instructions: Prepare one selector pack on the frozen `frontier_rebase/pr1394/train_gpt_human.py` base after the control-delta calibration pack has already established the downstream noise band. Materialize one training trace with the published SP8192 + GPTQ embeddings + depth recurrence + MuonEq-R + SDClip stack unchanged. From that single trace, expose a bounded late checkpoint window anchored to the existing harness checkpoint cadence: the default final checkpoint, the immediately previous saved checkpoint, and the second previous saved checkpoint if present. Cross those checkpoints with a fixed export menu already supported by the current harness catalog for this lane: the published default export path plus two non-default catalog export realizations already available to `search_harness` for downstream evaluation on pr1394. Build up to 6 candidate slots from this crossed grid, keeping 2 controls as exact repeats of the published default final-checkpoint/default-export pair. Do not retrain per candidate; all candidates must reuse the same trace and differ only in checkpoint choice and export recipe. Emit per-candidate metadata that records checkpoint identifier, export recipe identifier, downstream `score_bpb`, pre-quant `val_bpb`, and artifact bytes so the pack can be ranked strictly by downstream paired-control delta.
  composition_tags: same_trace, artifact_selection, post_train, deployment_touchpoint
  incompatible_with: none
  mutate_after_survival: expand_late_checkpoint_window, expand_fixed_export_menu, selector_robustness_across_seeds
- H01: trajectory_003/trajectory_003_h3 / trajectory_003_h3_r2 (Checkpoint-Only Sweep Under Frozen Default Export)
  family_group: checkpoint_export_selector
  rationale: Run immediately after control-delta calibration because it is executable now, lane-aligned, and tests a first-order searchable object without inventing new training machinery.
  instructions: Prepare a selector pack that isolates checkpoint choice while freezing export choice to the published default export path. Run one frozen pr1394 training trace with the published base stack unchanged. Enumerate a bounded late-checkpoint window from that same trace using the checkpoints already emitted by the harness near the end of training. Materialize up to 6 total candidate slots as concrete checkpoint picks from that window, always exported with the same default export recipe used by the published baseline. Keep 2 control slots as exact repeats of the default final checkpoint plus default export pair. The purpose is to falsify the invariant that the last checkpoint is automatically the best downstream artifact even when export stays fixed. Record the full downstream exact-eval result for each checkpoint candidate and keep the pack purely post-train: no per-candidate training edits, no export-method edits, no lane mixing.
  composition_tags: same_trace, artifact_selection, post_train, checkpoint_selector
  incompatible_with: none
  mutate_after_survival: expand_late_checkpoint_window, denser_checkpoint_sampling, selector_robustness_across_seeds

Control principles:
- Keep the published SP8192 + GPTQ embeddings + depth recurrence + MuonEq-R + SDClip stack frozen for all family admission comparisons.
- Include 2 controls in every executable pack; for calibration packs use replay-equivalent control clones to estimate pack-local noise.
- Never let a pre-quant improvement override a downstream-negative result.
- Reserve 2 slots for exact control repeats of the published default final checkpoint plus default export path.
- Use one shared training trace for every control and novel candidate in the pack.
- Reject any pack materialization that changes tokenizer, architecture, optimizer stack, dataset contract, or training budget.
- Use 2 exact controls with the default final checkpoint and default export path.
- All non-control candidates must differ from control only by checkpoint identity.
- Use the same saved training trace and the same evaluator/export path for every slot.

Admission principles:
- Run `trajectory_004/trajectory_004_h1` first and reuse its delta law for every later executable solo pack.
- Promote only families that beat the paired-control band on downstream `score_bpb` with repeated directional support.
- Keep `trajectory_004/trajectory_004_h3` and `trajectory_004/trajectory_004_h4` deferred until their required primitives exist.
- Primary admission is downstream delta versus the mean of the 2 controls on `quantized_sliding_window` `score_bpb`.
- A realization is positive only if at least one non-default checkpoint/export pair beats both controls by more than the calibrated control band.
- Do not promote a winner if its advantage appears only in pre-quant `val_bpb` while downstream `score_bpb` is neutral or worse.
- A checkpoint-only realization survives if a non-final late checkpoint beats both controls on downstream `score_bpb` beyond the calibrated control spread.
- If downstream ordering matches control noise and no non-final checkpoint separates, treat this realization as falsified rather than retuning offsets immediately.
- Do not backfill with export tweaks inside this realization; keep checkpoint choice isolated.

Metric principles:
- Primary metric is downstream `score_bpb` from `quantized_sliding_window` when present.
- Use pre-quant `val_bpb` only as secondary evidence for diagnosis or early kill guards.
- Use artifact size only as a tie-breaker after downstream and pre-quant evidence already agree.
- Primary metric: downstream `score_bpb` from `quantized_sliding_window` when present.
- Secondary metric: pre-quant `val_bpb` for diagnosis only.
- Tie-breaker: smaller artifact bytes only after downstream and pre-quant evidence agree.
- Primary metric: downstream `score_bpb` from `quantized_sliding_window`.
- Secondary metric: pre-quant `val_bpb` at the selected checkpoint, used only to interpret disagreement.
- Tie-breaker: artifact bytes if two checkpoints are downstream-indistinguishable.

Implementation principles:
- Be exact about catalog coverage: executable-now means current harness support; otherwise leave the family blocked with explicit primitive backlog.
- Keep probes minimal and lane-aligned: same-trace selector tests for selector families, mid-training probes for handoff programs, pairwise packs for branch portfolios.
- When a blocked family becomes runnable, start with its smallest decisive probe rather than a widened mutation slate.
- Use only catalog-supported checkpoint loading and export/eval modes already available in the current search harness.
- Keep the checkpoint window bounded and late; do not widen beyond the last few saved checkpoints in this first probe.
- Store candidate descriptors as explicit `(checkpoint_id, export_recipe_id)` pairs so later mutation stays on the searchable object rather than on prose labels.
- Prefer checkpoints aligned to existing validation/save cadence so no new runtime primitive is needed.
- Use explicit checkpoint offsets or ids, not scalar threshold tuning, to define the candidate set.
- This realization is the smallest decisive probe for the checkpoint half of the family.

Global campaign instruction:
Use `ncycle/pr1394_enigma_cycle.json` against the frozen `frontier_rebase/pr1394/train_gpt_human.py` base, with `frontier_rebase/pr1394/README.md` and `frontier_rebase/pr1394/submission.json` as the base contract. Execute solo packs first, always with 2 controls and up to 6 candidate slots, judge admission and promotion only on downstream `quantized_sliding_window` `score_bpb`, and record pack-local deltas to paired controls before promoting any family.

Hard execution rules:
- Preserve the requested slot ids exactly so downstream promotion can attribute outcomes.
- Do not substitute filler candidates if one realization cannot be mapped honestly.
- Prefer fewer executable candidates over invented or mixed-lane fillers.
- Treat this as one campaign pack, not a whole-strategy rewrite.
```
