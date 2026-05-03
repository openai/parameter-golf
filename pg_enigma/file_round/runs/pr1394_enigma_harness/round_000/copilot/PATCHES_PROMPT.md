# Patch Slate Copilot Prompt

Read the markdown brief plus Step 0 artifact, then materialize the final patch slate directly to files.

## Required reads
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/ROUND_BRIEF.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/step_0/STEP_0.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/FAMILY_INDEX.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/focus_files`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/evidence_files`

## Required writes
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/controls/C0.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/controls/C1.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/candidates/H0.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/candidates/H1.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/candidates/H2.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/candidates/H3.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/candidates/H4.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/candidates/H5.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/PATCH_INDEX.md`

## Family-first rule
- Use only families with positive candidate budget from `FAMILY_INDEX.md`.
- Respect the family budgets when assigning `H0`..`H5`.
- Represent at least 3 distinct source families across the six candidates.
- Do not invent a candidate that has no surviving family lineage.

## Candidate file contract
- Write one markdown file for each of `H0`..`H5`.
- Each candidate file must include: title, source family, mechanism, target surface, broken predicate, score path trace, operator claim, cheap signal, preserved contracts, likely failure mode, and an `Exact diff` section with a fenced ```diff block.
- Use a `## Source family` section that names exactly one surviving `F#` lineage.
- If a candidate is a selector or tournament, the diff must persist the selected winner as the final exported artifact and rerun the canonical final eval labels on that winner.
- If a candidate changes export encoding or recipe state, the deserialize/read path must be updated symmetrically so the chosen artifact can actually be loaded.
- Candidate diffs should be first-order and local enough to execute, but large enough to change program behavior meaningfully.
- Stay honest about why the diff should matter; no env churn, no pure literal retuning, no fake weirdness.

## Control file contract
- Write one markdown file for each of `C0` and `C1`.
- Controls must be explicit anchors for the tournament and should not introduce novel mechanisms.
- Controls may describe no-op/baseline behavior or replay behavior, but they must still state the intended comparison role.

## Patch index contract
- `PATCH_INDEX.md` must list all eight files, one-line rationale for each, and the intended tournament reading order.

## Output rules
- Markdown only. No JSON.
- Do not print the patch contents to stdout; write the files and return only a short completion note.
