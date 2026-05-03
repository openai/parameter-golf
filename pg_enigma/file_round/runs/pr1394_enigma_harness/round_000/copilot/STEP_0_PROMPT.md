# Step 0 Copilot Prompt

Read the markdown brief and round snapshots, then write the Step 0 artifact directly to disk.

## Required reads
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/ROUND_BRIEF.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/focus_files`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/evidence_files`

## Required write
- Create or update `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/step_0/STEP_0.md`

## File contract
Write markdown only. Use these sections exactly:
1. `# Step 0`
2. `## Measurement contract`
3. `## Score path to the deployed metric`
4. `## Mutation map`
5. `## Broken-predicate shortlist`
6. `## Final candidate lanes`
7. `## Controls`
8. `## Kill list`
9. `## Family-stage instructions`

## Candidate/Control contract
- The final candidate lanes section must name exactly `H0`, `H1`, `H2`, `H3`, `H4`, and `H5` as seed lanes, not final patch files.
- The controls section must name exactly `C0` and `C1`.
- Each candidate lane must state: target surface, target predicate, why baseline truth is false, score path trace, cheap signal, and likely failure mode.
- The family stage will regroup or split these seed lanes into `F0`..`F7` families before any final patch slate is written.
- Controls must be explicit anchors, not disguised candidates.
- Do not include diffs yet.
- Do not print the full artifact to stdout; write the file and return only a short completion note.
