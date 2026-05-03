# Agent Instructions

Work in the repository root. Do not modify unrelated source files.
Your task is to prepare generation `2` so `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/pack_handoffs/search_cycles/pr1394_enigma_6p2_pr1394_enigma_harness_r000/gen_002/bundle` exists and is ready to run on the GPU setup later.

## Inputs to read first
- Request: `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/pack_handoffs/search_cycles/pr1394_enigma_6p2_pr1394_enigma_harness_r000/gen_002/agent/request.json`
- Family catalog: `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/pack_handoffs/search_cycles/pr1394_enigma_6p2_pr1394_enigma_harness_r000/gen_002/agent/catalog.json`
- Postmortem memory: `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/pack_handoffs/search_cycles/pr1394_enigma_6p2_pr1394_enigma_harness_r000/gen_002/agent/postmortem_memory.json`
- Compiled schema: `parameter-golf/search_harness_compiled_generation_schema.json`
- Verifier schema: `parameter-golf/search_harness_verifier_report_schema.json`
- Compiled example: `parameter-golf/search_harness_compiled_generation_example.json`
- Focus-file snapshots: `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/pack_handoffs/search_cycles/pr1394_enigma_6p2_pr1394_enigma_harness_r000/gen_002/agent/focus_files`

## Focus repo files
- `frontier_rebase/pr1394/train_gpt_human.py`
- `frontier_rebase/pr1394/README.md`
- `frontier_rebase/pr1394/submission.json`
- `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/SEARCH_HANDOFF.md`
- `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/campaign.json`
- `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/pack_queue.json`
- `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/pack_handoffs/selector_pack_002_SEARCH_HANDOFF.md`
- `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/family_dossiers/trajectory_003_trajectory_003_h3.json`
- `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/family_compiles/trajectory_004_trajectory_004_h4.json`

## Required outputs to create
- `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/pack_handoffs/search_cycles/pr1394_enigma_6p2_pr1394_enigma_harness_r000/gen_002/compiled_generation.json`
- `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/pack_handoffs/search_cycles/pr1394_enigma_6p2_pr1394_enigma_harness_r000/gen_002/track_spec.json` (same content as compiled_generation.json is acceptable)
- `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/pack_handoffs/search_cycles/pr1394_enigma_6p2_pr1394_enigma_harness_r000/gen_002/verifier_report.json`

## Hard rules
1. Use only executable families from the family catalog.
2. Keep controls unpatched.
3. Use the fixed frozen base `frontier_rebase/pr1394/train_gpt_human.py` for every control and candidate.
4. Keep candidates first-order; do not create compounds.
5. Use at most one candidate per family_group.
6. BASE_DEFAULT_ENV in the request is the shared frozen env contract. Leave slot env empty unless a non-frozen override is truly needed, and never change these frozen keys: ['DATA_DIR', 'VOCAB_SIZE'].
7. If a single base cannot support the full slot budget, return fewer candidates instead of mixing bases or near-duplicate families.
8. Use postmortem memory from previous runs to avoid repeating mistakes.
9. If the user instructions ask for something impossible, choose the nearest executable pack that still obeys the frozen-base rules and explain the gap in metadata plus verifier warnings.

## After writing the JSON files
Run: `python3 search_harness.py run-generation --config parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/pack_handoffs/selector_pack_002_search_harness_config.json --generation 2 --dry-run`

## Success condition
- `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/pack_handoffs/search_cycles/pr1394_enigma_6p2_pr1394_enigma_harness_r000/gen_002/bundle/bundle_manifest.json` exists
- slot folders exist under `parameter-golf/pg_enigma/runs/pr1394_enigma_harness/round_000/pack_handoffs/search_cycles/pr1394_enigma_6p2_pr1394_enigma_harness_r000/gen_002/bundle/slots/`
- the generation folder is ready for later remote GPU execution

## User instructions
Use pg_enigma round 0 pack selector_pack_002 as the execution contract.
Pack kind: selector_pack
Lane: selector
Phase window: post_train

Create exactly these control slots and keep them unpatched:
- C0
- C1

Create candidate slots using these exact slot ids:
- H00: trajectory_003/trajectory_003_h3 / trajectory_003_h3_r3 (Export-Menu Sweep On Frozen Default Checkpoint)
  family_group: checkpoint_export_selector
  rationale: Run immediately after control-delta calibration because it is executable now, lane-aligned, and tests a first-order searchable object without inventing new training machinery.
  instructions: Prepare a selector pack that isolates export choice while freezing the checkpoint to the published default final checkpoint from one pr1394 training trace. Use the same frozen base stack and the same final checkpoint across the whole pack. Materialize a bounded export menu consisting of the published default export recipe plus up to 5 non-default export realizations that are already in the current search harness catalog for downstream evaluation on pr1394. Each candidate slot is one explicit export recipe applied to the same final checkpoint. Keep 2 controls as exact repeats of the default final checkpoint/default export pair. Evaluate every exported artifact on the same downstream `quantized_sliding_window` lane and record both downstream `score_bpb` and artifact bytes. The purpose is to falsify the invariant that export is just a passive tail step once the training trace is fixed.
  composition_tags: same_trace, artifact_selection, post_train, deployment_touchpoint, export_selector
  incompatible_with: none
  mutate_after_survival: expand_fixed_export_menu, selector_robustness_across_seeds, checkpoint_export_cross_after_solo_support

Control principles:
- Keep the published SP8192 + GPTQ embeddings + depth recurrence + MuonEq-R + SDClip stack frozen for all family admission comparisons.
- Include 2 controls in every executable pack; for calibration packs use replay-equivalent control clones to estimate pack-local noise.
- Never let a pre-quant improvement override a downstream-negative result.
- Reserve 2 exact control slots using the default final checkpoint and default export recipe.
- Freeze checkpoint identity for every non-control candidate; only export recipe may vary.
- Use the same downstream evaluator configuration for all slots.

Admission principles:
- Run `trajectory_004/trajectory_004_h1` first and reuse its delta law for every later executable solo pack.
- Promote only families that beat the paired-control band on downstream `score_bpb` with repeated directional support.
- Keep `trajectory_004/trajectory_004_h3` and `trajectory_004/trajectory_004_h4` deferred until their required primitives exist.
- A realization is positive only if a non-default export recipe beats both controls on downstream `score_bpb` beyond the calibrated control band.
- If an export recipe only improves size while downstream `score_bpb` is flat or worse, do not admit it.
- Do not mix in alternate checkpoints inside this realization; keep export choice as the only varying axis.

Metric principles:
- Primary metric is downstream `score_bpb` from `quantized_sliding_window` when present.
- Use pre-quant `val_bpb` only as secondary evidence for diagnosis or early kill guards.
- Use artifact size only as a tie-breaker after downstream and pre-quant evidence already agree.
- Primary metric: downstream `score_bpb` from `quantized_sliding_window`.
- Secondary metric: pre-quant `val_bpb` is expected to stay constant and should be logged mainly as a sanity check.
- Tie-breaker: artifact bytes among downstream-positive exports.

Implementation principles:
- Be exact about catalog coverage: executable-now means current harness support; otherwise leave the family blocked with explicit primitive backlog.
- Keep probes minimal and lane-aligned: same-trace selector tests for selector families, mid-training probes for handoff programs, pairwise packs for branch portfolios.
- When a blocked family becomes runnable, start with its smallest decisive probe rather than a widened mutation slate.
- Use only export/eval recipes already exposed by the current harness catalog; no new quantization primitive should be invented here.
- Treat each export recipe as an explicit named candidate so later mutation can expand the menu cleanly.
- Because checkpoint is frozen, any downstream win directly supports export-path selection as a first-order object.

Global campaign instruction:
Use `ncycle/pr1394_enigma_cycle.json` against the frozen `frontier_rebase/pr1394/train_gpt_human.py` base, with `frontier_rebase/pr1394/README.md` and `frontier_rebase/pr1394/submission.json` as the base contract. Execute solo packs first, always with 2 controls and up to 6 candidate slots, judge admission and promotion only on downstream `quantized_sliding_window` `score_bpb`, and record pack-local deltas to paired controls before promoting any family.

Hard execution rules:
- Preserve the requested slot ids exactly so downstream promotion can attribute outcomes.
- Do not substitute filler candidates if one realization cannot be mapped honestly.
- Prefer fewer executable candidates over invented or mixed-lane fillers.
- Treat this as one campaign pack, not a whole-strategy rewrite.

Finish by leaving the generation folder populated and dry-run materialized.
