---
title: feat: Parameter Golf Strategy Roadmap
type: feat
status: active
date: 2026-03-20
---

# feat: Parameter Golf Strategy Roadmap

## Overview

This plan turns the current strategy list into a staged campaign for Parameter Golf. The goal is not to pick a single “winner” up front, but to create a disciplined sequence of experiments that starts with a trustworthy baseline, then explores architecture, compression, training, and tokenizer ideas in a way that is measurable and reproducible.

The repo already has the right scaffolding for this work: explicit autoresearch modes, persisted trial metadata, and backend-specific training scripts (see `autoresearch/run_search.py:63-100`, `README.md:93-150`, `autoresearch/program.md:1-81`, and `CLAUDE.md:33-55`). The missing piece is a clear roadmap for which strategy family to attack first and how to record the results.

## Problem Statement / Motivation

The current research notes list several promising but competing directions:

- architecture changes: depth recurrence, width-heavy presets, aggressive GQA, MLP expansion
- compression changes: QAT, outlier-aware quantization, low-rank factorization
- training improvements: Muon tuning, longer contexts, curriculum learning
- tokenizer and evaluation changes: larger vocabularies and related bpb trade-offs

Without a staged plan, these ideas will blur together and produce noisy results. The repo’s own docs emphasize the 16,000,000 byte artifact cap and `val_bpb` as the authoritative objective, so the strategy roadmap has to preserve that measurement discipline while still allowing exploratory campaigns (see `README.md:104-108`, `autoresearch/program.md:7-24`).

## Proposed Solution

Use a phased roadmap with explicit gates:

1. **Phase 0 — Unblock and measure**
   - Run a real MLX baseline and verify the local data state.
   - Confirm whether one shard is enough for meaningful sweeps or whether the full download path is required.
   - Establish a reproducible baseline `val_bpb` and artifact size.

2. **Phase 1 — Architecture presets**
   - Add named preset families for `recurrent_deep`, `wide_shallow`, `aggressive_gqa`, and `large_vocab`.
   - Keep the preset-first workflow aligned with `autoresearch/run_search.py`’s existing `preset` mode and the docs’ recommended usage ladder.

3. **Phase 2 — Compression strategy track**
   - Add `layer_sharing` as the first code-mutation path for depth recurrence.
   - Add `qat_noise` as the first quantization-aware training experiment.
   - Keep `outlier-aware quant` and `low-rank factorization` as follow-up candidates if the earlier compression work shows promise.

4. **Phase 3 — Training optimization sweeps**
   - Sweep Muon-related hyperparameters.
   - Test longer sequence lengths where the artifact budget allows.
   - Keep curriculum learning as a later follow-up unless it can be expressed cleanly in the current env-var workflow.

5. **Phase 4 — Tokenizer and evaluation experiments**
   - Add a `large_vocab` campaign path for 2048/4096 vocab candidates.
   - Track the byte-per-byte tradeoff carefully, because the added vocabulary may help compression but can also increase artifact size.

6. **Phase 5 — Autoresearch campaigns**
   - Seed evolution from successful phase-1 and phase-2 trials.
   - Use the persisted trial JSON as the canonical lineage source.
   - Keep candidate scripts in `logs/autoresearch/workbench/` and never mutate the baseline training files in place.

## Why This Approach

This is the simplest plan that preserves signal quality:

- It starts with a baseline, which prevents us from optimizing against stale or invalid local results.
- It explores low-risk preset changes before high-risk compression or tokenizer work.
- It keeps code mutation separate from preset sweeps, which reduces confounding.
- It uses the repo’s existing four-mode autoresearch architecture rather than inventing a new workflow (see `docs/plans/2026-03-18-feat-autoresearch-search-modes-plan.md:32-72` and `autoresearch/run_search.py:765-925`).

Alternatives considered:

- **One giant sweep across all knobs** — rejected because it would blur architecture, compression, and tokenizer effects into one signal.
- **Jump straight to QAT or tokenizer changes** — rejected because these are harder to debug and need a stable baseline first.
- **Only preset work, no code mutation** — rejected because depth recurrence and quantization ideas need dedicated script variants to evaluate honestly.

## Technical Considerations

- Keep the search inputs env-var driven; `train_gpt.py` and `train_gpt_mlx.py` already expect configuration from environment variables rather than CLI flags (`CLAUDE.md:43-55`).
- Preserve the artifact cap and `final_int8_zlib_roundtrip` as the authoritative score gate.
- Keep backend parity: each strategy should have a clear path through both MLX and CUDA where practical.
- Avoid conflating research phases in one trial; record one strategy family per trial when possible.
- Keep new strategy names aligned with the current four-mode runner vocabulary and the docs’ usage ladder.

## System-Wide Impact

### Interaction Graph

`just` recipes invoke `autoresearch/run_search.py`, which selects a mode, generates a candidate config or workbench copy, launches `train_gpt.py` or `train_gpt_mlx.py`, parses metrics, and writes trial metadata to `logs/autoresearch/` (see `README.md:145-150`, `autoresearch/program.md:73-81`, `CLAUDE.md:49-55`).

For this roadmap:

- `preset` mode triggers new named architecture families.
- `code` mode is the path for `layer_sharing` and `qat_noise`.
- `evolution` mode should consume successful `trial/*.json` artifacts as lineage inputs.
- `best_config.json` remains the current best snapshot for resume behavior.

### Error Propagation

The main failure modes are:

- invalid preset values or unsupported env vars
- subprocess crashes in MLX/CUDA training
- parse failures when a run emits no final metrics
- over-limit artifacts that should not update the best result
- backend mismatch when resuming from prior artifacts

The plan should keep these failures isolated: a bad trial should still be recorded, but it should not corrupt best-result selection or population loading.

### State Lifecycle Risks

- `logs/autoresearch/results.tsv` can become misleading if a strategy family is renamed without updating the docs.
- `best_config.json` can go stale if backend-specific state is reused incorrectly.
- `logs/autoresearch/workbench/` must stay disposable and clearly owned by code-mutation runs.
- Larger vocab or QAT experiments may change artifact size faster than `val_bpb`, so both metrics need to be tracked together.

### API Surface Parity

Everything exposed to contributors should stay in sync:

- CLI modes in `autoresearch/run_search.py`
- `just` recipes in `justfile`
- usage examples in `README.md`
- policy notes in `autoresearch/program.md`
- operator guidance in `CLAUDE.md`

### Integration Test Scenarios

- Baseline MLX run completes and records a valid `val_bpb` and artifact size.
- A `recurrent_deep` or `wide_shallow` preset writes the expected trial metadata and stays under the artifact cap.
- `layer_sharing` writes candidate scripts only in `logs/autoresearch/workbench/`.
- `evolution --resume` seeds from prior successful trial JSON instead of starting from scratch.
- `large_vocab` either succeeds with the expected tokenizer data or fails early with a clear message if the tokenizer assets are missing.

## Acceptance Criteria

### Functional Requirements

- [ ] Phase 0 baseline is reproducible on MLX and records a real score, not a crash.
- [ ] Data availability is verified and documented before broader sweeps begin.
- [ ] `recurrent_deep`, `wide_shallow`, `aggressive_gqa`, and `large_vocab` are represented as named strategy families.
- [ ] `layer_sharing` and `qat_noise` exist as explicit code-mutation campaign paths.
- [ ] Evolution can be seeded from successful phase-1/phase-2 artifacts.
- [ ] The roadmap explicitly includes `outlier-aware quant` and `low-rank factorization` as follow-up exploratory tracks, even if they are not first-wave campaigns.

### Non-Functional Requirements

- [ ] Every trial must remain under the 16,000,000 byte artifact cap.
- [ ] Each experiment must preserve a single source of truth for `val_bpb`.
- [ ] Trial metadata must remain sufficient for resume, lineage, and provenance.
- [ ] The workflow should stay reproducible across MLX and CUDA backends where feasible.

### Quality Gates

- [ ] Static validation passes on the runner and training scripts.
- [ ] CLI examples in docs map to real commands.
- [ ] New strategy names are reflected in the docs and use the same vocabulary everywhere.
- [ ] The plan does not require in-place edits to `train_gpt.py` or `train_gpt_mlx.py` beyond the specific strategy support being added.

## Success Metrics

- At least one successful, reproducible MLX baseline is recorded.
- At least one successful trial exists for each of the primary architecture families.
- At least one successful code-mutation trial exists for depth recurrence and QAT.
- At least one tokenizer experiment is run with a documented byte trade-off.
- Successful trials are discoverable via `results.tsv`, `best_config.json`, and `trials/*.json` without manual reconstruction.

## Dependencies & Prerequisites

- Working MLX and/or CUDA environments.
- The existing four-mode autoresearch harness.
- `uv` and `just` workflows already used by the repository.
- Sufficient FineWeb shards for meaningful baseline and sweep runs.
- Tokenizer assets for any large-vocab campaign.

## Risk Analysis & Mitigation

- **Risk: strategy confounding** — Mitigation: one strategy family per campaign, preserved seeds, and consistent metadata.
- **Risk: data scarcity** — Mitigation: verify shard availability before claiming a result is meaningful.
- **Risk: artifact bloat** — Mitigation: keep `val_bpb` and byte size together in every decision.
- **Risk: code-mutation drift** — Mitigation: keep all generated variants in `logs/autoresearch/workbench/` and never edit baseline scripts in place.
- **Risk: QAT overreach** — Mitigation: treat QAT as a separate phase with its own success criteria instead of mixing it into architecture sweeps.

## Documentation Plan

- Update `README.md` if the roadmap changes the recommended usage order or adds new named presets.
- Update `autoresearch/program.md` if the search policy or mode ordering changes.
- Keep `CLAUDE.md` examples aligned with the commands contributors should actually run.
- If new mutation names become standard, document them in the autoresearch section so they are discoverable.

## Sources & References

### Internal References

- `README.md:93-150` — current autoresearch modes, scoring rules, and usage ladder.
- `autoresearch/program.md:1-81` — current search policy and mode descriptions.
- `autoresearch/run_search.py:63-100` — current search dimensions and backend choice space.
- `autoresearch/run_search.py:765-925` — mode dispatch and persistence behavior.
- `docs/plans/2026-03-18-feat-autoresearch-search-modes-plan.md:32-72, 181-199, 267-279, 391-400, 460-472` — prior plan and the gaps it identified.
- `CLAUDE.md:33-55` — current command surface and scoring pipeline summary.

### Institutional Learnings

- No `docs/solutions/` directory was present in this repository, so no institutional learnings were available to carry forward.

### Related Work

- The current leaderboard reference in `README.md:28-35` anchors the existing baseline score.
- Existing `just` commands already cover random, preset, evolution, and code modes; the roadmap should extend those, not replace them.

