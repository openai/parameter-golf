# Findings

## Main conclusion

The repo keeps rediscovering the same useful abstraction under different names:

- `hm2` = bootstrap -> handoff -> receiver
- `stage3_2` = bounded controller
- `stage3_4` = shared trunk -> late branch portfolio
- `stage3_5` = event-triggered branch tournament

Those are not four separate ideas. They are one search strategy: **search over small state-conditioned programs**, not over isolated patch packs.

The repeated mistake is that the harness still evaluates most work as stage-specific folders, prose-only hypotheses, or static patch slates. So it never compounds the one abstraction that actually looks general.

## Directory-level findings

| Dir | What it contributes | Specific mistake |
| --- | --- | --- |
| `hailmary/` | Best wide map of attack surfaces and high-upside families. | The **materialized runs** are still on `SP1024`, even though the source `run_configs.json` and docs now describe an `SP8192` strong base. Example: `hailmary/run_configs.json` defaults to `VOCAB_SIZE=8192`, but `hailmary/runs/tournament_20260411_001130/.../H2_exact_overlap_eval/config.json` still has `VOCAB_SIZE=1024`, `DATA_PATH=/data/...sp1024`, `TOKENIZER_PATH=/data/...1024_bpe.model`. |
| `hm2/` | First stage that clearly says the search unit is the **handoff policy**, not the patch. | It promises reusable evidence (`phase_diagnostics.json`) in `hm2/README.md`, but the repo contains no `train.log`, `summary.json`, or `phase_diagnostics.json` under `hm2/`. The idea is good, but the feedback loop never closed. |
| `stage3_1/` | Strongest admission bar for hypotheses; export-only lane is the right kind of isolation. | The stage has almost no run evidence in the repo. Current config/code point toward a real export-only bakeoff, but without artifacts the lane-honesty claim is still mostly architectural intent, not learned evidence. |
| `stage3_2/` | Best formal search substrate in the repo: bounded controller DSL, candidate metadata, canonicalization, mutation operators. | It still has **no result artifacts** in the repo, and the actual mutation code is simpler than the strategy docs. `controller_mutations.py` uses a fixed `60/30/10` numeric/wiring/structural split, while `evolution_strategy.md` argues for an annealed schedule by generation. |
| `stage3_3/` | Best library of cheap state features and phase signals (`train_loss_slope`, `step_avg_ms`, `update_norm`, etc.). | It is effectively noteware. It never became part of the active controller/search system, so the best feature work is stranded outside the runner that could use it. |
| `stage3_4/` | Introduces portfolio search over late finishers instead of one committed late path. | No logs, no summaries, no learning loop. Also, it treats branching as its own stage instead of as a reusable primitive that should compose with controllers and checkpoint selection. |
| `stage3_5/` | Closest thing to a general Enigma-like harness: trigger + branch programs + export-state portfolio. | Again, no result artifacts. And the branch DSL is still hand-authored presets in `stage3_5/patches.py` (`expand_finisher(...)`), not a searched grammar like `stage3_2`'s controller spec. |
| `stage3/` (reference) | The only place with enough logs to show what the runner actually does under pressure. | It proves the harness can screen and summarize, but it also shows the main failure modes directly: single-metric ranking across different causal lanes, stale plan/config drift, and orchestration breaks (`stage3/stage3_run.log` ends in `KeyError: 'final_single'`). |

## Specific mistakes

### 1. We are still searching the wrong unit

The most valuable objects in the repo are **transition programs**, not patches:

- `hm2` says the unit is the handoff policy.
- `stage3_2` says the unit is a bounded controller spec.
- `stage3_5` says the unit is a trigger plus branch-program portfolio.

But the actual harness still keeps falling back to:

- static patch packs
- stage-local slates
- prose-first hypothesis catalogs

That keeps the search near *"which local code diff helps?"* instead of *"which execution program wins under constraints?"*

### 2. Conceptual rebases are outrunning executed rebases

`hailmary` is the clearest example.

- Source docs/config now describe an `SP8192` strong local base.
- Actual April 11 run artifacts still use `SP1024`.

That means the project is often rebasing **on paper** before rebasing the runnable experiment surface. The result is false confidence: the writeup is aimed at one frontier while the jobs are still measuring another.

### 3. Our anchors are sometimes invalid, so rankings become fiction

The worst example is `hailmary/runs/tournament_20260411_001130/adhoc/screen/summary.json`.

Two nominal controls with the same seed and nearly identical materialized env:

- `H0_moonshot_control_base`: `217` steps, `829.69 ms`, `2.5957` post-quant BPB
- `H1_moonshot_control_repeat`: `431` steps, `418.46 ms`, `1.9951` post-quant BPB

That is not a small noise floor. That is a broken anchor. If the control repeat is ~`0.60` BPB better than the base control, then every candidate compared only to `H0` is being ranked against the wrong reference.

This is the most concrete sign that the harness is still vulnerable to scheduler/load/launch effects that swamp the actual hypothesis.

### 4. Lane isolation is still not enforced where it matters most

`hailmary` has a candidate literally called `exact_overlap_eval`, but its materialized config is still `runner_mode: "train"` and its run spends the whole `90s` wallclock training before export/eval:

- `hailmary/runs/tournament_20260411_001130/.../H2_exact_overlap_eval/config.json`
- `hailmary/runs/tournament_20260411_001130/.../H2_exact_overlap_eval/train.log`

So an eval-policy idea is still being tested as a full training run, not as a same-checkpoint eval bakeoff.

This is a repeated mistake across the repo: **export/eval hypotheses are not isolated enough from training**, so attribution stays muddy.

### 5. Export-heavy candidates are dying because the budget is not partitioned by lane

`hailmary` second-wave runs `M6` and `M7` both make it through training, then die in GPTQ calibration:

- `gptq:calibrating with 256 batches...`
- immediate `SIGTERM`

Files:

- `hailmary/runs/tournament_20260411_001130/moonshot_second_wave/screen/M6_moonshot_full_gptq_plus_xsa_all/train.log`
- `hailmary/runs/tournament_20260411_001130/moonshot_second_wave/screen/M7_moonshot_full_gptq_xsa_all_ema/train.log`

That means the runner is effectively asking some candidates to spend the *same* screen budget on:

1. training,
2. export,
3. calibration,
4. quantization,
5. eval.

For export-heavy candidates, that makes the screen measure "can this finish in time?" more than "is this a good export idea?"

### 6. We keep designing stages that never become learning systems

`hm2`, `stage3_1`, `stage3_2`, `stage3_3`, `stage3_4`, and `stage3_5` all have strong design docs, but the repo has **no `train.log` / `summary.json` / `phase_diagnostics.json` artifacts** under those directories.

In practice that means:

- the repo preserves hypothesis spaces,
- but not the actual evidence needed to retire families, compose winners, or estimate noise.

So the harness is over-producing search spaces and under-producing **search memory**.

### 7. We mutate within hand-picked families, but we do not search family structure itself

`stage3_2` is the best attempt at real evolutionary search, but it still starts from a human-seeded family list and mostly explores local neighborhoods around those seeds:

- `controller_library.py` defines the seed families.
- `evolve_stage3_2.py` generates children from those seeds.
- `controller_mutations.py` mutates magnitudes, wiring, and structure inside that bounded space.

That is good local search.

What is still missing is **family-level meta-search**:

- when to retire a family,
- when to compose two families,
- when to split a family,
- when to import a feature library (for example `stage3_3`) into the active controller grammar.

Right now the repo is much better at mutating a family than at deciding which families deserve mutation.

### 8. We keep separating naturally composable primitives into different stages

The repo already has the pieces of one unified search grammar:

- state features from `stage3_3`
- controllers from `stage3_2`
- portfolio finishers from `stage3_4`
- trigger + portfolio selection from `stage3_5`
- handoff logic from `hm2`

But instead of treating those as composable primitives in one harness, they are split into separate stage folders with separate runners.

That makes it much harder to ask the question that actually matters:

> does a good controller + a good branch portfolio + good checkpoint selection beat any one of them alone?

### 9. Our patching substrate is still too brittle for a general harness

The stage patch libraries are mostly exact string-replacement systems.

That has already bitten the repo:

- `stage3/review_feedback.md` documents the broken `label_smoothing` patch after root-script drift.
- The later-stage patches still depend on exact snippet matches in copied training scripts.

This is manageable for a one-off research stage. It is the wrong substrate for a general Enigma-style optimizer that is supposed to survive code evolution.

## What the logs actually say

### `stage3/` is the clearest honest warning

The `stage3` logs show what a real screen can teach:

- `H3_nuclear_norm_regularization` and `H6_zloss_plus_nuclear_norm` are not subtle losers.
- They are catastrophic on both quality and throughput.

From `stage3/runs/stage3_original/screen/summary.json`:

- control `R0A`: `1.8825` post-quant BPB at `500.13 ms`
- `H3`: `3.2821` post-quant BPB at `1843.06 ms`
- `H6`: `3.2886` post-quant BPB at `1827.97 ms`

That is exactly the kind of result that should retire a family, not just mark a slot as a loser.

At the same time, `stage3/stage3_run.log` ends with:

- `KeyError: 'final_single'`

So even the one stage with real logs still shows that orchestration correctness is not yet reliable enough.

## Reverse-engineered search strategy

The strongest general strategy already hiding in the repo is:

## **Phase-Trigger-Portfolio (PTP) search**

Treat each candidate as a **small execution program**, not as a diff.

### Candidate schema

```text
candidate:
  base_variant
  signals[]              # cheap observable state
  phase_boundaries[]     # early/mid/late or equivalent
  triggers[]             # threshold or event rules
  actions[]              # what changes when triggered
  portfolio[]            # alternate finishers/export modes/backends
  selector               # how the winning state/artifact is chosen
  dominant_metric
  expected_horizon
  early_signal
  kill_rule
  composition_role
```

### Why this is the right abstraction

- `hm2` already uses **phase -> handoff -> receiver**
- `stage3_2` already uses **signals -> gates -> actions -> snapshot/pulse**
- `stage3_4` already uses **shared trunk -> portfolio of finishers -> best branch wins**
- `stage3_5` already uses **trigger -> branch programs -> export-state portfolio**

That is one grammar. The repo should stop treating them as different stage species.

## What to build next

### 1. One unified bounded DSL

Merge these into one search object:

- `stage3_3` feature library
- `stage3_2` controller/gate/action DSL
- `stage3_4` portfolio/branch primitive
- `stage3_5` trigger + selector primitive
- `hm2` handoff primitive

Code patches should become **leaf actions** inside this DSL, not the top-level search unit.

### 2. Mandatory evidence files per run

No stage should count as "ran" unless it emits:

- `manifest.json` - candidate spec as executed
- `metrics.json` - lane-specific metrics
- `events.json` - triggers, handoffs, branch choices
- `selection.json` - why the final state/artifact won
- `failure.json` - declared failure mode hit or not

That would immediately fix the current problem where many directories contain only copied `train_gpt.py` and `config.json`.

### 3. Family-level search, not just candidate-level search

Add explicit family operators:

- retire family
- split family
- compose families
- rebase family onto new base

Right now only `stage3_2` has serious candidate mutation. The harness still lacks **family mutation**.

### 4. Lane-specific budgeting

Do not give all candidates the same wallclock contract.

- training-heavy candidates need train budget
- export-heavy candidates need export/calibration budget
- eval-only candidates should run on the same checkpoint

Without that, export/eval ideas keep losing for scheduling reasons instead of causal reasons.

### 5. Control validation before ranking

Before promoting any candidate, require:

- repeated control stability,
- matched lane control,
- no anchor drift like the `hailmary` `H0` vs `H1` split.

If the control repeat is not stable, the harness should refuse to rank the pack.

## Short version

The repo's best hidden insight is this:

> the winning search object is not a patch.  
> it is a bounded state-conditioned program with triggers, handoffs, branches, and selection rules.

`hm2`, `stage3_2`, and `stage3_5` are all pointing at that.

The main mistakes are:

1. still searching patch packs instead of programs,
2. rebasing the docs faster than the executed artifacts,
3. trusting unstable controls,
4. mixing training/export/eval lanes,
5. creating stages that do not emit reusable evidence,
6. keeping naturally composable primitives in separate folders instead of one DSL.

If the goal is an Enigma-like general harness, the next step is not another new stage.

It is to unify the repo around **Phase-Trigger-Portfolio search** and make every experiment emit machine-readable evidence that can retire, mutate, or compose families.
