---
title: feat: Complete autoresearch search modes workflow
type: feat
status: completed
date: 2026-03-18
---

# feat: Complete autoresearch search modes workflow

## Enhancement Summary

**Deepened on:** 2026-03-18
**Plan mode:** in-place expansion
**Primary grounding:** repository source, current `justfile`, current docs, and spec text

### Key improvements added

1. Split the work into a true gap-closure plan instead of a greenfield implementation plan.
2. Added a concrete metadata contract so TSV, best-result JSON, trial JSON, and workbench artifacts stay aligned.
3. Added a verification matrix that separates no-cost CLI parsing from optional live MLX smoke validation.
4. Added explicit `justfile` command design and documentation rollout requirements.
5. Added a sharper risk model around backwards compatibility, artifact churn, and population seeding behavior.

### New considerations discovered

- The requested modes, presets, lineage fields, and workbench-copy behavior are already partially implemented in the runner; the main risk is incomplete productization, not missing primitives.
- `results.tsv` currently omits parent lineage and candidate script path even though trial JSON already stores richer metadata, which creates a misleading split-brain logging model.
- The plan needs to treat `best.config` carefully because it currently carries a synthetic `DESCRIPTION` field, which is useful for humans but noisy as future machine input.

## Overview

The repository already contains a substantial partial implementation of explicit autoresearch modes in [autoresearch/run_search.py](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py). The remaining work is to turn that partial implementation into a coherent, documented, and verifiable workflow that contributors can actually use for local MLX iteration and challenge-oriented CUDA runs.

This plan treats the work as a completion and hardening pass rather than a greenfield build. The main deliverable is a stable search runner with clear mode semantics for `random`, `preset`, `evolution`, and `code`, plus matching `just` entrypoints, persisted metadata, and updated documentation.

### Research Insights

**Best Practices:**
- Preserve working orchestration seams when a script is already doing the right high-level job; strengthen invariants before introducing structural refactors.
- Treat CLI, task runner, and docs as one user-facing surface. A mode that exists only in source code is not really shipped.

**Implementation Details:**
- The runner already has stable top-level seams for mode entrypoints, persistence, and candidate-script creation.
- The deepest value now comes from clarifying contracts between those seams rather than adding new abstractions.

**Edge Cases:**
- A feature can appear “done” because the main file contains the right nouns, while still being operationally incomplete because entrypoints and docs lag behind.
- Seemingly harmless metadata changes can break `--resume` or pollute future search inputs.

## Problem Statement

The spec asks for four explicit search modes, architecture presets, persisted per-trial metadata, evolutionary seeding, dedicated-copy code mutation, expanded logging, new `just` commands, and updated docs. Much of that now exists in code:

- Mode dispatch and CLI flags already exist in [autoresearch/run_search.py:730](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L730)
- Presets already exist for `cuda` and `mlx` in [autoresearch/run_search.py:76](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L76)
- Code mutations and workbench script generation already exist in [autoresearch/run_search.py:243](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L243) and [autoresearch/run_search.py:512](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L512)
- Trial JSON persistence and population loading already exist in [autoresearch/run_search.py:456](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L456) and [autoresearch/run_search.py:475](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L475)

The missing piece is consistency across code, task runner, and docs. `justfile` still only exposes legacy random-style commands in [justfile:38](/Users/cortex-air/Developer/parameter-golf/justfile#L38), while README and program docs still describe a simpler single-loop workflow in [README.md:87](/Users/cortex-air/Developer/parameter-golf/README.md#L87) and [autoresearch/program.md:47](/Users/cortex-air/Developer/parameter-golf/autoresearch/program.md#L47).

### Gap Inventory

The plan should explicitly close these gaps:

1. **Public interface gap**
   - CLI supports four modes, but `justfile` only exposes the original basic search flow.
2. **Documentation gap**
   - README and `autoresearch/program.md` still describe the older lightweight loop and do not explain when to use each mode.
3. **Metadata visibility gap**
   - Trial JSON has richer metadata than TSV, so a contributor reading only `results.tsv` cannot reconstruct lineage or candidate-script provenance.
4. **Verification gap**
   - The spec explicitly asks for compile-time and dry CLI validation, but the current workflow docs do not institutionalize those checks.
5. **Terminology gap**
   - The public names requested in the spec should be reconciled against current preset names such as `depth_first`, `width_first`, `compact_context`, `small_fast`, and `balanced`.

## Proposed Solution

Keep the current runner architecture and finish the workflow around it instead of rewriting the file. The implementation should preserve the existing `TrialResult`, preset dictionaries, mutation library, and per-mode entrypoints, then tighten mode behavior, eliminate rough edges, and expose the supported workflow consistently through `justfile` and docs.

The plan should explicitly frame the work as:

1. Verify and normalize runner behavior for all four modes.
2. Make metadata and resume/evolution semantics trustworthy.
3. Add first-class command entrypoints for every supported mode.
4. Update documentation so contributors know which mode to use and when.
5. Add sanity validation that proves the interface parses cleanly before expensive training runs.

### Research Insights

**Best Practices:**
- Make the persisted artifact model explicit before extending search logic further.
- Prefer mode-specific clarity over “smart” automatic behavior; search workflows are easier to trust when users can see which strategy is running.

**Implementation Details:**
- The solution should define one canonical source for each concern:
  - runner logic: `autoresearch/run_search.py`
  - public entrypoints: `justfile`
  - workflow guidance: `README.md`
  - policy and search intent: `autoresearch/program.md`
  - machine-readable run history: `logs/autoresearch/trials/*.json`

**Edge Cases:**
- `--resume` must not silently use stale or backend-incompatible state.
- Code mode must not leave ambiguity about whether a result came from a baseline script or a workbench copy.

## Technical Approach

### Architecture

Use `autoresearch/run_search.py` as the single orchestration surface. The runner already has the right top-level decomposition:

- shared config normalization and artifact estimation
- explicit per-mode functions
- common `run_trial(...)`
- JSON per-trial persistence
- workbench copies for code mutation

The work should preserve that structure and avoid moving search logic into the training scripts. The baseline training files remain source-of-truth execution targets, and code-mutation mode should continue to operate only on generated copies under `logs/autoresearch/workbench/`.

### Research Insights

**Best Practices:**
- Keep orchestration near the experiment bookkeeping instead of leaking it into model-training entrypoints.
- Prefer append-only trial artifacts over derived state when enabling future search strategies like resume and evolution.

**Implementation Details:**
- `TrialResult` should remain the central persistence contract.
- `ensure_dirs()`, `append_result()`, `load_best()`, and `load_population()` should be treated as a single persistence subsystem, not as unrelated helpers.
- Candidate-script generation in `create_candidate_script(...)` should remain the only code path that produces mutable training-script copies.

**Metadata Contract To Enforce**

Every persisted trial should make the following reconstructable without reading raw logs:

- run identity: `run_id`, backend, mode, status
- search strategy inputs: config, preset, code mutation, parents
- artifact provenance: baseline script path or generated workbench script path
- evaluation outputs: `val_bpb`, `val_loss`, total bytes, quantized model bytes, parameter count
- operator context: description/delta, elapsed seconds, log path

This contract already mostly exists in JSON. The implementation should decide whether to:

- expand TSV to include more of it, or
- explicitly document TSV as a short summary and JSON as the source of truth

Either choice is valid; ambiguity is not.

### Implementation Phases

#### Phase 1: Audit and tighten runner semantics

- Review each mode path in [autoresearch/run_search.py:621](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L621), [autoresearch/run_search.py:639](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L639), [autoresearch/run_search.py:657](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L657), and [autoresearch/run_search.py:709](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L709).
- Confirm the intended semantics for `--preset`, `--population`, `--code-mutation`, `--resume`, and `--baseline-first`.
- Fix gaps where the code technically supports a feature but the behavior is not yet robust. Likely areas:
  - whether `results.tsv` should include lineage-related columns instead of hiding them only in JSON
  - whether estimate-only candidate scripts in code mode should be cleaned up or deliberately retained
  - whether `load_population()` should filter or rank by mode/preset when building the evolutionary pool
  - whether `best.config` carrying a synthetic `DESCRIPTION` field is desirable input for future normalization
- Keep all metadata additions backwards-compatible with existing `best_config.json` and trial JSON artifacts where possible.

**Detailed execution checklist**

- Read every helper from `normalize_config(...)` through `main()` as one flow, not as isolated functions.
- Write down intended semantics for each flag before changing code:
  - `--mode`
  - `--preset`
  - `--population`
  - `--code-mutation`
  - `--resume`
  - `--baseline-first`
- Decide whether synthetic seed results in evolution mode should remain ephemeral or become visible persisted artifacts.
- Decide how skipped trials should be surfaced when artifact estimates reject them before execution.
- Decide whether code mode should keep `estimate_*` workbench copies for auditability or clean them to reduce churn.

**Deliverable of Phase 1**

A short internal decision record, either as comments in the code or as plan-aligned implementation notes, that states the intended meaning of each mode and each metadata surface.

#### Phase 2: Finalize presets and metadata contract

- Keep the challenge-oriented preset families already defined in [autoresearch/run_search.py:76](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L76), but rename or rebalance them only if the current names do not match the requested public interface.
- Decide and document the canonical preset vocabulary across both backends:
  - baseline
  - depth-heavy or equivalent
  - width-heavy or equivalent
  - compact-context / high-throughput or equivalent
- Make the per-trial JSON artifact the canonical resume/evolution source.
- Expand TSV and JSON output so every completed trial unambiguously captures:
  - `mode`
  - `preset`
  - `code_mutation`
  - `parents`
  - candidate script path
  - human-readable delta/description

**Preset normalization tasks**

- Map current preset names to public-facing descriptions:
  - `depth_first` → depth-heavy family
  - `width_first` → width-heavy family
  - `compact_context` → compact-context / high-throughput family
  - `small_fast` and `balanced` → local-iteration MLX families
- Decide whether to rename the underlying keys or keep internal keys stable and document the mapping.
- Ensure the named preset interface behaves the same on both backends, even if the actual hyperparameter bundles differ.

**Metadata design tasks**

- Decide whether `results.tsv` should gain columns for:
  - `parents`
  - `train_script_path`
  - `train_script_bytes`
- If TSV remains intentionally compact, document that:
  - TSV is for fast scanning
  - per-trial JSON is for resume/evolution/debugging
- Ensure `best_config.json` is either:
  - a full `TrialResult`, or
  - a clearly documented reduced schema

**Backward-compatibility rule**

No change in this phase should require deleting existing `logs/autoresearch/trials/*.json` files just to restore normal operation.

#### Phase 3: Expose commands in `justfile`

- Preserve existing entrypoints at [justfile:38](/Users/cortex-air/Developer/parameter-golf/justfile#L38) through [justfile:45](/Users/cortex-air/Developer/parameter-golf/justfile#L45) for continuity.
- Add explicit commands for MLX and CUDA variants of:
  - preset mode
  - evolution mode
  - code mode
- Prefer names that match the docs and CLI, for example:
  - `autoresearch-preset-mlx`
  - `autoresearch-preset-cuda`
  - `autoresearch-evolution-mlx`
  - `autoresearch-evolution-cuda`
  - `autoresearch-code-mlx`
  - `autoresearch-code-cuda`
- Include parameters for `trials`, `seed`, `nproc`, and optional named preset / mutation where appropriate.

**Command design guidance**

- Keep existing commands as friendly defaults:
  - `autoresearch-mlx`
  - `autoresearch-cuda`
  - `autoresearch-resume`
- Add new explicit commands rather than overloading the existing recipes with too many optional branches.
- Prefer one recipe per intent instead of one recipe with many positional parameters; this keeps README examples readable.

**Minimum command matrix**

- MLX:
  - default random search
  - preset search
  - evolution search
  - code-mutation search
- CUDA:
  - default random search
  - preset search
  - evolution search
  - code-mutation search

**Acceptance note**

If the new recipes require more than one optional selector each, consider named variables in the recipe signature so command examples remain self-explanatory.

#### Phase 4: Update user-facing documentation

- Rewrite the README autoresearch section in [README.md:87](/Users/cortex-air/Developer/parameter-golf/README.md#L87) so it describes the four-mode model instead of the older single-loop story.
- Update [autoresearch/program.md:1](/Users/cortex-air/Developer/parameter-golf/autoresearch/program.md#L1) to document search policy by mode, expected outputs, and recommended mode order.
- Document a simple usage ladder:
  - MLX preset or random for fast local iteration
  - MLX evolution after a few seed runs exist
  - CUDA preset/random to validate promising families remotely
  - CUDA evolution and code mutation for more aggressive search

**Documentation outline to add**

- What each mode is optimizing for
- When to use it
- Required inputs and useful optional flags
- What artifacts it writes
- How resume interacts with prior results
- Which modes are best for:
  - first local exploration
  - building a preset baseline
  - mining prior successes
  - trying structural code perturbations

**Documentation quality bar**

- Every command shown in README must exist verbatim in `justfile` or as direct CLI syntax in the runner.
- README should remain newcomer-oriented.
- `autoresearch/program.md` should remain operator-oriented and policy-focused.

#### Phase 5: Sanity validation

- Run `python3 -m py_compile autoresearch/run_search.py`.
- Run dry CLI parse checks for each mode, for example with `--help` plus one command per mode using minimal flags that avoid real training.
- If possible without expensive runtime, execute one tiny MLX parse/smoke invocation to ensure mode routing, log directory creation, and artifact writing still work.

**Verification matrix**

1. **Static validation**
   - `python3 -m py_compile autoresearch/run_search.py`
2. **CLI surface validation**
   - `--mode random`
   - `--mode preset --preset baseline`
   - `--mode evolution --population <N>`
   - `--mode code --code-mutation <name>`
3. **Task-runner validation**
   - each new `just` recipe expands to the expected runner command
4. **Documentation validation**
   - each documented example maps to a real command
5. **Optional live validation**
   - one intentionally tiny MLX invocation if environment and time allow

**Verification rule**

Dry parse validation is required. Live training validation is recommended but should stay optional in the plan because contributor environments may not have MLX or CUDA ready at plan time.

## Alternative Approaches Considered

### 1. Rewrite the runner around subcommands

This would make the CLI cleaner, but it is unnecessary now. The current `--mode` dispatch is already present and good enough for the repository’s size.

### 2. Split each mode into separate modules

This would reduce file length, but the repo currently uses a flat, script-oriented structure. A modular split is extra surface area without immediate payoff.

### 3. Treat the feature as already complete

This would ignore the mismatch between code, task runner, and docs. The repository would still be confusing for contributors and difficult to validate.

### Research Insights

**Best Practices:**
- When a feature is partially shipped, the highest-leverage planning move is to distinguish “missing implementation” from “missing operational completeness.”
- Plans should explicitly reject large refactors when the real problem is packaging, public interface, and verification.

## System-Wide Impact

### Interaction Graph

`justfile` commands call `autoresearch/run_search.py`, which selects a mode, generates a candidate config or workbench copy, launches `train_gpt.py` or `train_gpt_mlx.py`, parses emitted metrics, and persists TSV/JSON logs under `logs/autoresearch/`. In code mode, the runner also writes candidate script copies into `logs/autoresearch/workbench/` before dispatching training.

### Error & Failure Propagation

Training or parse failures currently collapse into `crash` or `parse_error:*` statuses in [autoresearch/run_search.py:585](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L585). The plan should preserve that coarse status model unless there is a clear need for more states, but docs should explain how failures appear in logs and trial JSON.

**Additional failure modes to cover**

- invalid preset names or code-mutation names
- backend mismatch between current run and resumed best artifact
- shape-invalid candidates that require repeated resampling
- artifact estimate rejection loops that could silently skip many trials
- mutation-target drift if baseline training scripts change and textual mutation anchors stop matching

### State Lifecycle Risks

The main risk is metadata drift: `results.tsv`, `best_config.json`, per-trial JSON, and workbench copies can diverge if one output format changes without the others. Another risk is accidental dependence on mutated workbench scripts outside code mode. The implementation should keep the baseline scripts immutable and treat workbench copies as disposable run artifacts.

**State-management expectations**

- `best_config.json` should never point to a non-reconstructable candidate.
- Trial JSON should remain sufficient to rebuild an evolutionary pool after process restart.
- Workbench scripts should be treated as run artifacts, not hand-edited assets.
- If workbench cleanup is introduced, cleanup must never delete the script referenced by a persisted successful trial without some replacement provenance path.

### API Surface Parity

Every mode exposed in the CLI must have a matching `just` wrapper and documentation example. The current mismatch between `run_search.py` and `justfile` is the clearest parity gap.

### Integration Test Scenarios

- Running `--mode preset --preset baseline` writes consistent TSV and JSON metadata.
- Running `--mode evolution --resume` with existing trials seeds from prior artifacts instead of ignoring them.
- Running `--mode code --code-mutation gelu_mlp` writes a workbench script and leaves `train_gpt.py` / `train_gpt_mlx.py` untouched.
- Switching `--backend` between `mlx` and `cuda` does not reuse incompatible `best_config.json` state.
- Over-limit or parse-error runs remain recorded without corrupting best-result selection.

**Additional scenarios**

- Named preset selection on MLX and CUDA yields distinct backend-appropriate configs under the same logical preset family.
- Population size capping removes weaker candidates but never discards the current best result.
- `--baseline-first` runs only when no valid incumbent exists for the active backend.
- Code mode with a mutation name that no longer matches the training script fails clearly rather than silently generating an invalid experiment.

## Acceptance Criteria

### Functional Requirements

- [x] `autoresearch/run_search.py` cleanly supports `random`, `preset`, `evolution`, and `code` as explicit public modes.
- [x] Both `cuda` and `mlx` expose challenge-oriented presets and allow either named selection or sampling across presets.
- [x] Evolution mode can seed from persisted per-trial artifacts and maintain a bounded top-K population.
- [x] Code mode writes candidate training scripts only under `logs/autoresearch/workbench/` and never edits baseline training files.
- [x] Per-trial metadata persists enough information to support resume, lineage tracing, and population loading.
- [x] `justfile` contains dedicated MLX and CUDA commands for preset, evolution, and code modes while preserving existing simple commands.
- [x] `README.md` and `autoresearch/program.md` explain all modes, recommended usage order, and backend guidance.
- [x] The plan’s implementation does not require editing `train_gpt.py` or `train_gpt_mlx.py` to support the new search workflow.
- [x] Resume behavior is explicitly defined for backend mismatch, missing best artifact, and population reconstruction from trial JSON.

### Non-Functional Requirements

- [x] The runner remains script-oriented and easy to inspect.
- [x] Metadata formats remain human-readable.
- [x] Dry validation does not require expensive training by default.
- [x] The logging model is understandable to a contributor reading only docs and artifact filenames.
- [x] New commands remain terse enough to be practical in README snippets and shell history.

### Quality Gates

- [x] `python3 -m py_compile autoresearch/run_search.py` passes.
- [x] One dry CLI parse check passes for each mode.
- [x] Documentation examples match actual CLI and `just` entrypoints.
- [x] At least one verification step confirms that code mode writes a generated script into `logs/autoresearch/workbench/`.
- [x] At least one verification step confirms that baseline scripts are unchanged after code-mode validation.

## Success Metrics

- Contributors can discover every supported mode from `justfile` and README without reading the runner source.
- Evolution and code-mode artifacts are sufficient to explain how a candidate was produced.
- A contributor can choose the right local-vs-remote mode from docs in under a minute.
- Contributors can tell which artifact to inspect first:
  - quick summary: `results.tsv`
  - best run snapshot: `best_config.json`
  - full provenance: `trials/*.json`
  - generated candidate code: `workbench/`

## Dependencies & Prerequisites

- Existing runner structure in [autoresearch/run_search.py](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py)
- Existing `uv` + `just` workflow
- Working MLX and/or CUDA environments depending on validation depth

## Recommended Execution Order

1. Audit current mode semantics and record intended contracts.
2. Normalize preset naming and metadata expectations.
3. Tighten persistence and resume/evolution behavior.
4. Add explicit `justfile` commands for each supported mode.
5. Rewrite README and `autoresearch/program.md` around the new public workflow.
6. Run static and dry CLI validation.
7. Optionally run one tiny MLX live validation.

## Risk Analysis & Mitigation

- Risk: over-planning work that is already implemented.
  - Mitigation: start with a gap audit and only change behavior that is inconsistent or undocumented.
- Risk: docs advertise commands that do not exist.
  - Mitigation: update `justfile` before final documentation wording.
- Risk: resume/evolution semantics become brittle with legacy JSON artifacts.
  - Mitigation: preserve backwards-compatible reads or add explicit migration handling.
- Risk: richer TSV logging makes the summary file noisy or hard to scan.
  - Mitigation: decide explicitly whether TSV is compact summary or full lineage table and document that choice.
- Risk: code-mode estimate artifacts create filesystem clutter.
  - Mitigation: choose either retention-with-documentation or cleanup-with-safety rules; avoid accidental middle ground.
- Risk: public preset labels drift from actual backend-specific tuned values.
  - Mitigation: document logical preset families separately from exact per-backend configs.

## Implementation Notes For The Engineer

- Treat this as a completion pass, not a redesign project.
- Avoid splitting `run_search.py` unless a concrete defect forces it.
- Prefer adding a few well-named helpers over introducing classes or new modules.
- Keep plan, commands, and docs synchronized in the same change set.
- Verify artifact paths and filenames directly rather than assuming they are correct.

## Documentation Plan

- Update [README.md](/Users/cortex-air/Developer/parameter-golf/README.md) autoresearch usage and examples.
- Update [autoresearch/program.md](/Users/cortex-air/Developer/parameter-golf/autoresearch/program.md) with mode descriptions, policy, and usage order.
- Ensure command examples match new `justfile` recipes exactly.

### Documentation content checklist

- README:
  - short mode overview
  - recommended usage order
  - example `just` commands
  - artifact outputs
- `autoresearch/program.md`:
  - operator intent for each mode
  - mode-specific search policy
  - metadata and provenance expectations
  - resume/evolution behavior
- Optional inline help improvements in `argparse` descriptions if the CLI output is currently too terse.

## Sources & References

### Internal References

- Existing presets and mutation library: [autoresearch/run_search.py:76](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L76)
- Trial metadata and persistence: [autoresearch/run_search.py:303](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L303)
- Mode implementations and CLI flags: [autoresearch/run_search.py:621](/Users/cortex-air/Developer/parameter-golf/autoresearch/run_search.py#L621)
- Current public commands: [justfile:38](/Users/cortex-air/Developer/parameter-golf/justfile#L38)
- Current README autoresearch docs: [README.md:87](/Users/cortex-air/Developer/parameter-golf/README.md#L87)
- Current program doc: [autoresearch/program.md:1](/Users/cortex-air/Developer/parameter-golf/autoresearch/program.md#L1)

### Institutional Learnings

- No matching `docs/solutions/` entries were present in this repository at planning time.

### External References

- No external research was required. The codebase already contains the relevant implementation patterns and the work is primarily a repo-internal completion and documentation pass.
