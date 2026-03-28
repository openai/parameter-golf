# Worklog Templates

Use these only when the repo does not already provide a stronger template.

## Repo Discovery Checklist

- Check whether the repo already has `docs/`, `build-logs/`, `runbooks/`, `plans/`, `tickets`, or memory-log files.
- Check for `_template.md`, `index.md`, or dated file naming conventions.
- Check whether evidence artifacts are tracked, curated, or ignored by default.
- Prefer updating an existing surface before creating a parallel one.

## Build Log Template

```md
# <Topic> - <YYYY-MM-DD>

## Objective

<What this tranche is trying to accomplish>

## Changes Landed

- <concrete change>
- <concrete change>

## Validation

- `<command>`
- `<command>`

## Evidence

- `<artifact path>`
- `<artifact path>`

## Outcome

- <what improved, regressed, or remains blocked>

## Next Action

- <next step>
```

## Memory Log Template

```md
## Session <YYYY-MM-DD HH:MM> - <Short title>

- Tranche/Phase: <phase>
- Goal: <goal>

### Work Completed

- <change>
- <change>

### Findings

- <what was learned>
- <what failed or remains risky>

### Validation

- `<command>`

### Evidence

- `<artifact path>`

### Next Action

- <next step>
```

## Experiment Note Template

```md
## Experiment - <Short title>

- Hypothesis: <belief being tested>
- Change: <what was changed>
- Result: <pass/fail/mixed>
- Evidence: <paths, metrics, screenshots, logs>
- Decision: <adopt, revise, abandon, revisit later>
```

## What To Capture

- The hypothesis or root-cause theory.
- The exact files or systems touched.
- The commands actually run.
- The artifact paths that prove the result.
- The trade-off or reason a path was rejected.
