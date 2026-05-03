# Markdown File Round Brief

## Identity
- Cycle: `pr1394_enigma_harness`
- Round: `0`
- Target script: `frontier_rebase/pr1394/train_gpt_human.py`
- Goal: minimize credible val_bpb

## User instructions
Build a fresh Step 0, then a family slate, then a six-candidate, two-control patch slate for the frontier target.

## Metric contract
- Primary metric: `downstream score_bpb from quantized_sliding_window when present`
- Secondary metrics:
- `pre-quant val_bpb`
- `control spread`
- `artifact size only as a tie-breaker`

## Hard constraints
- This lane writes markdown files only. Do not emit JSON artifacts.
- Run the stages in order: Step 0 -> family slate -> final patch slate.
- Families come before patches: do not jump from Step 0 directly to final candidate diffs.
- The final slate must contain exactly 6 candidate files (`H0`..`H5`) and 2 control files (`C0`, `C1`).
- The family slate must preserve multiple mechanism families and allocate the six candidate slots across surviving families.
- Preserve broken-predicate honesty: do not sell a candidate that solves an already-satisfied predicate on the score path.
- Prefer consequential program mutations over local knob nudges, env churn, or plausible-but-unfalsified narratives.

## Output contract
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/step_0/STEP_0.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/FAMILY_INDEX.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F0.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F1.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F2.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F3.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F4.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F5.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F6.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/families/F7.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/controls/C0.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/controls/C1.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/candidates/H0.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/candidates/H1.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/candidates/H2.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/candidates/H3.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/candidates/H4.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/candidates/H5.md`
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/PATCH_INDEX.md`

## Canonical repo files
- `frontier_rebase/pr1394/train_gpt_human.py`

## Focus snapshots for this round
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/focus_files/frontier_rebase/pr1394/train_gpt_human.py`

## Evidence snapshots for this round
- `parameter-golf/pg_enigma/file_round/runs/pr1394_enigma_harness/round_000/evidence_files/parameter-golf/findings.md`

## Reference docs
- `parameter-golf/findings.md`
- `parameter-golf/search_strategy_catalog.md`
- `parameter-golf/search_strategy_meta_prompt.md`
- `parameter-golf/search_verifier_codex.md`
- `parameter-golf/pg_enigma/self_check_rubric.md`
- `frontier_rebase/pr1394/README.md`
- `frontier_rebase/pr1394/submission.json`

## Objective notes
- The fixed frontier base is the published pr1394 SP8192/GPTQ/recurrence/MuonEq/SDClip stack.
- This lane exists to test the pg_enigma campaign workflow itself on the real pr1394 target while leaving the previously staged direct harness lane untouched.
- Executable-now families should target the current search_harness catalog honestly; blocked families should stay blocked instead of pretending to be runnable.

## Prior round context
_No prior round history in configured window._

## Prior postmortem context
_No prior postmortem context was available._
