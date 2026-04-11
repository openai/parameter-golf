# Stage 3.1 Review

Date: 2026-03-25

## Verdict

`stage3_1` does not fully pass the current bar.

It is better than the older narrow-helper stages on hypothesis distinctness, and the updated work did improve two things:

- there is now an explicit hypothesis-stage bar in [hypothesis_stage_bar.md]( nanoevolve/pgolf/parameter-golf/stage3_1/hypothesis_stage_bar.md)
- the config now distinguishes some `candidate` vs `support` roles and adds a dedicated export control slot

But it still fails two important requirements:

1. the lane structure is not actually implemented the way the config claims
2. the runnable search still mixes observability lanes in a way that weakens attribution

So the idea quality is partly there, but the experimental design is not yet honest enough.

## Hard Findings

### 1. `lane_b_bakeoff` is not actually export-only

The config claims Lane B runs on an existing checkpoint with no training:

- [`run_configs.json:3`]( nanoevolve/pgolf/parameter-golf/stage3_1/run_configs.json#L3)
- [`run_configs.json:45`]( nanoevolve/pgolf/parameter-golf/stage3_1/run_configs.json#L45)

But the orchestrator just launches the patched training script normally:

- [`orchestrate_stage3_1.py:530`]( nanoevolve/pgolf/parameter-golf/stage3_1/orchestrate_stage3_1.py#L530)

And the base script has no checkpoint-load / export-only path at all. It always constructs loaders, trains, then exports:

- [`base_train_gpt.py:1143`]( nanoevolve/pgolf/parameter-golf/stage3_1/base_train_gpt.py#L1143)
- [`base_train_gpt.py:1192`]( nanoevolve/pgolf/parameter-golf/stage3_1/base_train_gpt.py#L1192)
- [`base_train_gpt.py:1302`]( nanoevolve/pgolf/parameter-golf/stage3_1/base_train_gpt.py#L1302)

So the most important claimed advantage of the stage is currently false in implementation.

### 2. `--phase all` skips declared phases

The config defines:

- `lane_b_bakeoff`
- `composite`
- `decision`

at:

- [`run_configs.json:45`]( nanoevolve/pgolf/parameter-golf/stage3_1/run_configs.json#L45)
- [`run_configs.json:51`]( nanoevolve/pgolf/parameter-golf/stage3_1/run_configs.json#L51)
- [`run_configs.json:59`]( nanoevolve/pgolf/parameter-golf/stage3_1/run_configs.json#L59)

But `--phase all` only runs:

- `sanity`
- `screen`
- `final_single`
- optional `champion_8x`

at:

- [`orchestrate_stage3_1.py:558`]( nanoevolve/pgolf/parameter-golf/stage3_1/orchestrate_stage3_1.py#L558)

So the staged search is overstated relative to what the one-command path actually does.

### 3. The short screens mix lane A and lane B hypotheses

Both `sanity` and `screen` run:

- export-only hypotheses `H1-H3`
- training hypotheses `H4-H7`

in the same short packs:

- [`run_configs.json:35`]( nanoevolve/pgolf/parameter-golf/stage3_1/run_configs.json#L35)
- [`run_configs.json:40`]( nanoevolve/pgolf/parameter-golf/stage3_1/run_configs.json#L40)

That weakens attribution. Lane B ideas should ideally be compared on the same checkpoint, not mixed into the same early training tournament as lane A.

### 4. The new hypothesis-stage bar is not yet enforced by the runnable stage

The new bar requires each idea to specify:

- broken invariant
- expected impact with magnitude and horizon
- kill rule
- composition role
- code burden

at:

- [hypothesis_stage_bar.md:22]( nanoevolve/pgolf/parameter-golf/stage3_1/hypothesis_stage_bar.md#L22)
- [hypothesis_stage_bar.md:72]( nanoevolve/pgolf/parameter-golf/stage3_1/hypothesis_stage_bar.md#L72)
- [hypothesis_stage_bar.md:142]( nanoevolve/pgolf/parameter-golf/stage3_1/hypothesis_stage_bar.md#L142)
- [hypothesis_stage_bar.md:155]( nanoevolve/pgolf/parameter-golf/stage3_1/hypothesis_stage_bar.md#L155)
- [hypothesis_stage_bar.md:165]( nanoevolve/pgolf/parameter-golf/stage3_1/hypothesis_stage_bar.md#L165)

But the actual slot entries still do not encode most of those fields. For example, the updated slot schema has `why`, `validates`, `falsifies`, and `notes`, but not explicit fields for `broken_invariant`, `expected_horizon`, `kill_rule`, or `code_burden`:

- [run_configs.json:125]( nanoevolve/pgolf/parameter-golf/stage3_1/run_configs.json#L125)
- [run_configs.json:219]( nanoevolve/pgolf/parameter-golf/stage3_1/run_configs.json#L219)
- [run_configs.json:267]( nanoevolve/pgolf/parameter-golf/stage3_1/run_configs.json#L267)

So the new bar exists as policy text, but the runnable stage does not yet enforce it structurally.

## Bar Assessment

### What passes

These parts are genuinely stronger than older stages:

- The hypotheses are more distinct than typical optimizer-helper packs.
- The updated config now at least distinguishes some `support` ideas from lead `candidate` ideas.
- There is real coverage across:
  - export function
  - export policy
  - loss alignment
  - architecture allocation
  - deploy-alignment schedule
  - objective schedule
- `H6 quant_anneal` and `H7 staged_objective` are the closest to the current bar because they alter the process, not just a local coefficient.
- `H1-H3` are legitimate export-lane ideas, not just hyperparameter retunes.

### What still fails the bar

The stage still under-satisfies the new standard in four ways:

1. It does not break enough false invariants in the training process.
   Missing:
   - checkpoint selection
   - data-order staging
   - parameter-family specialization
   - context-budget staging

2. It over-relies on analogy naming.
   Some slots are genuinely mechanism-level, but others feel more like "cross-field framing on top of a local patch" than a truly radical implementation.

3. The lane split is described better than it is implemented.
   This is the biggest practical issue.

4. The new hypothesis-stage bar is documented, but not enforced by the config schema or runner.

5. The main one-command path still behaves like a classic screen/promote runner, not a fully lane-aware tournament.

## Promising Ideas

The best ideas here, independent of the implementation problems, are:

- `H6 quant_anneal`
- `H7 staged_objective`
- `H2 fisher_bit_allocation`
- `H1 companding`

Those are the ones most worth preserving if the stage is rebuilt.

The weakest ideas relative to the new bar are:

- `H4 byte_weighted_loss`
  It is defensible, but still closer to loss retuning than to a large broken-invariant mechanism.
- `H3 sparsify_5pct`
  It is interesting for compression, but as written it is more of a helper lane than a lead stage.

## Overall Judgment

If the question is "is this stage better than the older narrow packs?" the answer is yes.

If the question is "does this pass the current bar for initial ideas and stage design?" the answer is no, not yet.

The core reason is simple:

- the hypothesis families are partially wide enough
- the implementation and tournament structure are not yet honest enough about lanes

That means `stage3_1` is a promising draft stage, not yet a bar-clearing one.
