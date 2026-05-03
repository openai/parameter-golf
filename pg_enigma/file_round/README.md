# pg_enigma file_round

This is the **markdown-only** lane for `pg_enigma`.

It does **not** touch the existing `pg_enigma.py` workflow. Everything here lives under this folder.

## Goal

Turn one optimization brief into:

1. a **Step 0** markdown artifact that locks the measurement contract, score path, mutation map, broken predicates, seed lanes, and controls
2. a **family slate** that preserves multiple mechanism families, judges them, and allocates the six candidate slots across surviving families
3. a final markdown patch slate with:
   - `controls/C0.md`
   - `controls/C1.md`
   - `candidates/H0.md` .. `candidates/H5.md`
   - `PATCH_INDEX.md`

No JSON outputs are written by this lane.

## Commands

All commands live under:

```bash
python3 pg_enigma/file_round/file_round.py <command> ...
```

### 1. Prepare the round

This creates:

- `ROUND_BRIEF.md`
- copied `focus_files/`
- copied `evidence_files/`
- `copilot/STEP_0_PROMPT.md`
- `copilot/FAMILIES_PROMPT.md`
- `copilot/PATCHES_PROMPT.md`
- `ROUND_SUMMARY.md`
- `POSTMORTEM_CONTEXT.md` when a prior classic `pg_enigma` postmortem exists

Example:

```bash
python3 pg_enigma/file_round/file_round.py prepare-round \
  --config pg_enigma/pr1394_enigma_config.json \
  --round 0 \
  --instructions "Build a fresh Step 0 and then a six-candidate, two-control patch slate for the frontier target." \
  --repo-file frontier_rebase/pr1394/train_gpt_human.py \
  --evidence-file findings.md
```

### 2. Run Step 0

This asks Copilot to **write the artifact directly to file**:

```bash
python3 pg_enigma/file_round/file_round.py run-step0 \
  --config pg_enigma/pr1394_enigma_config.json \
  --round 0
```

Required output:

- `step_0/STEP_0.md`

### 3. Run the family slate

This asks Copilot to write the intermediate **family-first** layer:

```bash
python3 pg_enigma/file_round/file_round.py run-families \
  --config pg_enigma/pr1394_enigma_config.json \
  --round 0
```

Required outputs:

- `FAMILY_INDEX.md`
- `families/F0.md`
- `families/F1.md`
- `families/F2.md`
- `families/F3.md`
- `families/F4.md`
- `families/F5.md`
- `families/F6.md`
- `families/F7.md`

The family stage is where the harness:

- widens seed lanes into mechanism families
- marks each family `KEEP`, `REWRITE`, or `DROP`
- allocates the six candidate slots across surviving families
- prevents direct Step 0 -> final patch collapse

### 4. Run the patch slate

This asks Copilot to **write the final files directly**:

```bash
python3 pg_enigma/file_round/file_round.py run-patches \
  --config pg_enigma/pr1394_enigma_config.json \
  --round 0
```

Required outputs:

- `controls/C0.md`
- `controls/C1.md`
- `candidates/H0.md`
- `candidates/H1.md`
- `candidates/H2.md`
- `candidates/H3.md`
- `candidates/H4.md`
- `candidates/H5.md`
- `PATCH_INDEX.md`

These candidates must now be compiled **from the surviving families**, not directly from Step 0.

If the patch slate fails validation, the harness writes repair feedback under `copilot/`, regenerates a repair prompt, and reruns the patch stage instead of accepting the broken slate.

### 5. One-command flow

```bash
python3 pg_enigma/file_round/file_round.py run-round \
  --config pg_enigma/pr1394_enigma_config.json \
  --round 0 \
  --instructions "Build a fresh Step 0 and then a six-candidate, two-control patch slate for the frontier target." \
  --repo-file frontier_rebase/pr1394/train_gpt_human.py \
  --evidence-file findings.md
```

Resume from an already prepared round:

```bash
python3 pg_enigma/file_round/file_round.py run-round \
  --config pg_enigma/pr1394_enigma_config.json \
  --round 0 \
  --resume
```

## Workspace layout

Rounds are written under:

```text
pg_enigma/file_round/runs/<cycle_id>/round_000/
```

Typical contents:

```text
ROUND_BRIEF.md
POSTMORTEM_CONTEXT.md          # optional
ROUND_SUMMARY.md
focus_files/
evidence_files/
step_0/
  STEP_0.md
FAMILY_INDEX.md
families/
  F0.md
  F1.md
  F2.md
  F3.md
  F4.md
  F5.md
  F6.md
  F7.md
controls/
  C0.md
  C1.md
candidates/
  H0.md
  H1.md
  H2.md
  H3.md
  H4.md
  H5.md
PATCH_INDEX.md
copilot/
  STEP_0_PROMPT.md
  FAMILIES_PROMPT.md
  PATCHES_PROMPT.md
  PATCH_REPAIR_PROMPT.md         # written when patch repair is needed
  PATCH_VALIDATION_FEEDBACK.txt  # latest patch validation failure
  step0_stdout.log
  step0_stderr.log
  families_stdout.log
  families_stderr.log
  patches_stdout.log
  patches_stderr.log
  patch_repair_*_stdout.log
  patch_repair_*_stderr.log
```

## Design constraints

- markdown only
- no JSON outputs
- Copilot writes files directly
- Step 0 comes before family materialization
- families come before final patch materialization
- patch slate stays explicit: six candidates, two controls
- the six candidate slots are allocated across surviving families first
- patch validation enforces selector winner materialization and recipe read/write symmetry, with repair retries when needed
- prior classic `pg_enigma` postmortem can be carried in as markdown context when available
