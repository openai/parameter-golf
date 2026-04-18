# 2026-04-19 — Record-track kickoff + workflow scaffold

## Decision: going for the record track

Switching from the non-record (baseline-improvement) track to the **record track**. Target: beat SOTA 1.0810 bpb by 2026-04-30 (~11 days).

- Previous best (Exp 21): 1.1963 on 2×H100 40min (invalid format).
- Exp 24: replicated SOTA code pre-quant at 1.0985 on 2×H100, harness is sound. Moving on without further post-mortem.
- Real gap to close on 8×H100: ~0.02 bpb from straight replication + micro-optimizations.

## Decision: flatten `parameter-golf-upstream/` into `parameter-golf/`

The nested repo layout (outer `ai-workspace/projects/parameter-golf/` + inner `parameter-golf-upstream/` fork) was confusing. Flattened: the fork's `.git` now lives at `parameter-golf/`, all code + scaffold + notes are at the repo root. Outer `ai-workspace` will stop tracking `parameter-golf/*` (move to .gitignore separately).

## Decision: workflow scaffold (no training code changes)

Built a research-ops scaffold on a new `research` branch off `beating-sota`. Two session modes:

- **Research session** — slow, no pod, writes ideas / specs / evaluations / diary / code diffs on branches.
- **Execution session** — fast, pod live, runs one spec at a time, produces `runs/NNN-slug/` artifacts, stops pod.

Three-phase loop per idea: **spec → run → evaluate**, with file-based handoff (no inter-session chat).

### Scaffold created today
- `CLAUDE.md` — shared repo conventions
- `EXECUTION.md` — detailed execution protocol (hardware ladder, interview, preflight, artifact shape)
- `.claude/skills/research.md` — `/research` role activator
- `.claude/skills/execution.md` — `/execution` role activator
- `research/{ideas,specs,evaluations}/` — lifecycle dirs
- `runs/` — artifacts
- `research/specs/000-sota-replication.md` — first spec (baseline validation on 8×H100)
- `research/ideas/` seeded with 6 Stage 1/2 candidates: progressive-recurrence, disable-layer0-attn, swa-plus-ema, bigram-hash, layerwise-lr-decay, per-group-quant

### Conventions locked in
- **Numbering:** single linear counter, assigned at spec-freeze.
- **Multi-seed:** one spec, one dir per seed under `runs/NNN-slug/seed_XX/`.
- **Ledger:** one row per spec in `experiments.md`, written by research at evaluation time.
- **Code ownership:** research writes logic (on `exp/<slug>` branches, pinned commit); execution only does environmental fixes.
- **Artifacts:** small in-repo, checkpoints on NA-1 volume; in-repo `checkpoints.md` is a pointer.
- **Failed runs:** keep the number, `status: failed` in `final.json`.

## Next session

- First execution run: **spec 000** (SOTA replication, seed 42, 8×H100 NA-1, ~$3.50).
- Fill in the HEAD commit hash of the `research` branch into spec 000 before handoff.
- Confirm SP8192 training data exists on the NA-1 volume (open question in the spec).
