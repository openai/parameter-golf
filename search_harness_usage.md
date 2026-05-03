# Search Harness Usage

## Main commands

## Repo-ready config

For this repo, use:

```text
ncycle/my_cycle.json
```

That config is wired for:

- repo-root outputs in `search_cycles/`
- GPU-targeted torchrun entrypoints
- repo-specific extraction from `train.log`
- Codex retries and verifier gating
- survival-pack gating: fixed `stage3_3/base_train_gpt.py`, one env contract, one candidate per family group
- predicate-audit gating: each candidate must prove a broken predicate on the real baseline score path before admission
- remote paths matching the `/data/parameter-golf` cluster layout

### 1. Direct Codex loop

Use this when you want one command to ask Codex for an executable family pack, verify it, retry if needed, and leave a runnable generation folder.

```bash
python3 search_harness.py codex-generation \
  --config ncycle/my_cycle.json \
  --generation 0 \
  --repo-file stage3_3/base_train_gpt.py \
  --repo-file stage3_3/patches.py \
  --instructions-file my_brief.txt
```

Add `--execute --remote` if you want the same command to launch the resulting bundle on the GPU host after the folder is ready.

If the Codex preparation was interrupted, resume it from the stored `codex/request.json` and any existing attempt artifacts:

```bash
python3 search_harness.py codex-generation \
  --config ncycle/my_cycle.json \
  --generation 0 \
  --resume
```

`--resume` reuses the saved request and focus set, so do not pass new `--repo-file` values when resuming.

### 2. External-agent handoff

Use this when you want to prepare a folder that Copilot, Claude Code, or Codex can read and complete.

```bash
python3 search_harness.py prepare-agent-folder \
  --config ncycle/my_cycle.json \
  --generation 0 \
  --repo-file stage3_3/base_train_gpt.py \
  --repo-file stage3_3/patches.py \
  --instructions-file my_brief.txt
```

This writes:

```text
search_cycles/<cycle_id>/gen_000/agent/
  AGENT_INSTRUCTIONS.md
  AGENT_PROMPT.txt
  request.json
  catalog.json
  postmortem_memory.json
  focus_files/
  search_harness_compiled_generation_schema.json
  search_harness_verifier_report_schema.json
  search_harness_compiled_generation_example.json
```

You can then tell an external agent:

```text
Read search_cycles/<cycle_id>/gen_000/agent/AGENT_INSTRUCTIONS.md and complete the task fully.
```

The agent is instructed to create:

```text
search_cycles/<cycle_id>/gen_000/
  compiled_generation.json
  track_spec.json
  verifier_report.json
  bundle/
```

## Inputs

- `my_cycle.json`: global harness config
- `my_brief.txt`: what you want this generation to try
- `--repo-file ...`: repo files the agent should inspect first

## Outputs

- `compiled_generation.json`: exact slots to run
- `verifier_report.json`: PASS / PASS_WITH_WARNINGS / RETRY / FAIL
- `bundle/`: staged, patched, runnable generation folder
- `codex/` or `agent/`: audit trail, prompts, schemas, request context, and memory

## Postmortem memory

Both `codex-generation` and `prepare-agent-folder` include:

- prior generation summaries
- prior verifier outcomes
- reference excerpts from:
  - `findings.md`
  - `search_reset.md`
  - `search_reset_agent_prompt.md`
  - `search_verifier_codex.md`

## Where to review the logic

- `search_harness.py`
  - `build_postmortem_memory`
  - `prepare_generation_with_codex`
  - `build_agent_instructions`
  - `prepare_agent_folder`
- `search_harness_pgolf_extractor.py`
- `search_harness_usage.md` for the high-level flow
- `search_harness_compiled_generation_example.json` for the executable slot shape
