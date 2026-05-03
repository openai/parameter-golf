# Family index

## Proposed families

| Family | Title | Seed lineage | Verdict | Candidate budget |
| --- | --- | --- | --- | --- |
| F0 | Late checkpoint portfolio selector | H0 | REWRITE | 1 |
| F1 | Frozen-checkpoint export menu selector | H1 | KEEP | 2 |
| F2 | Recurrence handoff program | H2 | REWRITE | 1 |
| F3 | Optimizer phase-law controller | H3 | DROP | 0 |
| F4 | XSA late-depth program | H4 | DROP | 0 |
| F5 | Skip-correction controller | H4 | DROP | 0 |
| F6 | GPTQ clip-law and bit-allocation rewrite | H5 | KEEP | 2 |
| F7 | Export reserve / calibration scheduler | H5 | DROP | 0 |

## Surviving families

- `F0` survives because checkpoint identity is directly on the deployed score path, but it needs honest same-trace snapshot plumbing before it becomes a real selector family.
- `F1` survives as the cleanest executable-now post-train family: freeze training, vary only the export recipe, and rank on `quantized_sliding_window` against `C0` and `C1`.
- `F2` survives because loop activation is a real transition program, not a scalar threshold truth, but it must be rewritten as a handoff policy rather than a bare `ENABLE_LOOPING_AT` sweep.
- `F6` survives because the README and script both make GPTQ law first-order for deployed score and bytes, and it can be isolated on a frozen checkpoint.

## Candidate budget allocation

| Family | Budget | Why it gets slots |
| --- | --- | --- |
| F0 | 1 | One honest checkpoint-selector slot is enough until late-snapshot plumbing exists. |
| F1 | 2 | This is the highest-confidence executable-now family and should preserve multiple export-menu variants. |
| F2 | 1 | Keep one non-selector transition-program lineage alive without pretending the current harness can search it broadly. |
| F6 | 2 | Export-law changes are direct-to-score and deserve a second slot for materially different quantizer rewrites. |
| F3 | 0 | Dropped. |
| F4 | 0 | Dropped. |
| F5 | 0 | Dropped. |
| F7 | 0 | Dropped. |

Total candidate slots: `6`.

## Drop/rewrite rationale

- `F0` is `REWRITE`, not `KEEP`, because the current script exports only the final EMA state; patch stage must first materialize a small late-checkpoint portfolio from one identical trace.
- `F2` is `REWRITE`, not `KEEP`, because a real recurrence family needs a handoff program plus activation diagnostics, not a lone threshold nudge.
- `F3` is `DROP` because it currently degenerates into broad optimizer tuning with weak cheap falsifiers and no clean downstream attribution before export.
- `F4` is `DROP` because `xsa_last_n` is currently a static suffix-width knob with poor observability and no honest shortcut to deployed-score attribution.
- `F5` is `DROP` because skip-controller changes co-adapt with training and looping; without richer diagnostics the family is too entangled for a six-slot slate.
- `F7` is `DROP` because reserve-seconds budgeting is a harness-allocation issue that confounds training time with export completion rather than testing artifact quality directly.

## Patch-stage instructions

1. Keep `F0` and `F1` separate in the patch slate: checkpoint identity and export identity must not be merged into one mixed candidate.
2. Build `F0` candidates as same-trace late-checkpoint selectors only; freeze the default export recipe and score each snapshot with the deployed sliding evaluator.
3. Build `F1` and `F6` as frozen-checkpoint export-only candidates; do not let them borrow extra training budget or change score semantics.
4. Build `F2` as a transition-program family with explicit activation logic and activation-side diagnostics; reject candidates that are only scalar threshold sweeps in disguise.
5. Every surviving family must be compared to both `C0` and `C1`; if control spread covers the observed delta, mark the family unresolved rather than positive.
6. Do not spend any patch slots on `F3`, `F4`, `F5`, or `F7` in this round.
