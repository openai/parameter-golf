---
title: Parameter Golf Subagents
read_when:
  - You are spawning Codex subagents for repo exploration, experiment design, or run tooling.
  - You want consistent agent roles before parallel work starts.
---

# Parameter Golf subagents

Project-scoped agent files live in `.codex/agents/`.

Use these roles:

- `pg_repo_mapper`: trace training, validation, quantization, serialization, artifact bytes, submission flow.
- `pg_compression_researcher`: focus on post-quant delta, byte budget, packing, quantization-friendliness.
- `pg_experiment_designer`: design sweep matrices, run names, promotion gates, and success criteria.
- `pg_run_operator`: wire scripts, logs, and repeatable experiment plumbing.

Default fan-out for new work:

1. `pg_repo_mapper` first.
2. `pg_compression_researcher` in parallel if the task touches model bytes or post-quant score.
3. `pg_experiment_designer` when deciding what to run next.
4. `pg_run_operator` only after the plan is stable enough to implement.

Working norms:

- Explorers stay read-only and cite files plus env knobs.
- Implementation work stays small and reversible.
- Optimize for `final_int8_zlib_roundtrip` and total bytes, not only pre-quant `val_bpb`.
- Promote ideas from local smoke -> 1xH100 -> 8xH100 only when the previous gate is clearly positive.
