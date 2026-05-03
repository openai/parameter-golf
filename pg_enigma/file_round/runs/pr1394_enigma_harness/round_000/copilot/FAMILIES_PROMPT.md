# Family Slate Copilot Prompt

Read the brief plus Step 0 artifact, then write the family slate directly to disk.

## Required reads
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/ROUND_BRIEF.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/step_0/STEP_0.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/focus_files`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/evidence_files`

## Required writes
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/FAMILY_INDEX.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F0.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F1.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F2.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F3.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F4.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F5.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F6.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F7.md`

## Operating mode
- First act as a wild proposer: widen the mechanism space into families instead of jumping to final diffs.
- Then act as a ruthless empiricist: mark each family `KEEP`, `REWRITE`, or `DROP` by predicate truth, score-path relevance, and continuation value.
- The purpose of this stage is to preserve families, not collapse immediately to one winner.

## Family file contract
- Write one markdown file for each of `F0`..`F7`.
- Each family file must use these sections exactly:
  1. `# F? - ...`
  2. `## Title`
  3. `## Mechanism family`
  4. `## Search level`
  5. `## Operator family`
  6. `## Target surface`
  7. `## Broken predicate`
  8. `## Why baseline truth is false`
  9. `## Score path trace`
  10. `## Mutation grammar`
  11. `## Cheap falsifier`
  12. `## Preserved contracts`
  13. `## Verdict`
  14. `## Candidate budget`
  15. `## Family compile instructions`
  16. `## Likely failure mode`
- `## Verdict` must be exactly one of `KEEP`, `REWRITE`, or `DROP`.
- `## Candidate budget` must be exactly `0`, `1`, or `2`.
- A `DROP` family must have budget `0`.
- A `KEEP` or `REWRITE` family must have budget `1` or `2`.

## FAMILY_INDEX.md contract
- Use these sections exactly:
  1. `# Family index`
  2. `## Proposed families`
  3. `## Surviving families`
  4. `## Candidate budget allocation`
  5. `## Drop/rewrite rationale`
  6. `## Patch-stage instructions`
- The index must mention every family id `F0`..`F7`.
- Allocate exactly 6 total candidate slots across surviving families.
- Keep at least 3 families alive with positive budget so the final slate cannot collapse to one lineage.
- Do not write final candidate diffs in this stage.

## Output rules
- Markdown only. No JSON.
- Do not print the family contents to stdout; write the files and return only a short completion note.
