# Agent lab — commit conventions

## Format (Conventional Commits)

```
<type>(<scope>): <short imperative description>

[optional body]
```

- **Types:** `feat` (behavior or defaults change), `fix`, `docs`, `chore`, `refactor`, `test`.
- **Scope:** `agent-lab` (preferred) or `agent_lab`.

## Subject line

- Imperative mood: *“halve train batch”*, not *“halved”*.
- ~50–72 characters; no period at the end.

## Body (recommended for every experiment commit)

Include a block agents and humans can grep:

```
Exp: AL-YYYYMMDD-NNN
Parent: <7-char sha>
Hypothesis: <one sentence>
Metric: final_int8_ttt_lora (lower better) | or name if you switched
Result: keep | discard (with val_bpb if useful)
```

Example:

```
feat(agent-lab): default NUM_KV_HEADS=2

Exp: AL-20260328-002
Parent: df152a4
Hypothesis: GQA with 2 KV heads increases steps/min and improves TTT val_bpb
Metric: final_int8_ttt_lora
Result: keep (1.5921 vs baseline 1.6099)
```

## Historical commits

Older commits on `agent_lab/mar28` may use labels like `agent_lab E3:`; **new work** should follow this document. Experiment identity lives in **`experiments.tsv`** and commit bodies, not only the subject line.
