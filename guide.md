# Guide

## Goal

This repo now supports a simple workflow:

1. prepare a generation folder locally
2. let Codex / Claude Code / Copilot fill in the tournament pack
3. keep multiple prepared generations on disk while waiting for GPUs
4. run a prepared generation later from its own folder
5. optionally use SSH/rsync from the repo config

This guide documents the **current working flow**, including what is clean, what is automated, and what is still manual.

---

## Main files

### Primary config

Use:

```text
ncycle/my_cycle.json
```

Current important values:

- `cycle_id`: `pgolf_family_search`
- `workspace.root`: `../search_cycles`
- `admission.mode`: `survival`
- `admission.require_single_base`: `true`
- `admission.require_unique_family_groups`: `true`
- `admission.fixed_base_script`: `stage3_3/base_train_gpt.py`
- `base.defaults.VOCAB_SIZE`: `1024`
- `codex.model`: `gpt-5.4`
- outputs land under:

```text
search_cycles/pgolf_family_search/
```

### Main scripts

- `search_harness.py` — prepare generations, materialize bundles, execute tournaments, select survivors
- `search_harness_pgolf_extractor.py` — parse `train.log` into selector metrics

### Main prompt/doctrine docs

- `findings.md`
- `search_reset.md`
- `search_reset_agent_prompt.md`
- `search_verifier_codex.md`

These are included as postmortem / memory context when preparing agent packets.

---

## The clean mental model

Separate the workflow into **two phases**:

### 1. Prepare generation folders

No GPU needed.

You can create:

- `gen_000`
- `gen_001`
- `gen_002`

and let them sit on disk until hardware is available.

### 2. Execute a prepared generation later

When GPUs are free, run a prepared `bundle/` from that generation.

This is the part the clean-folder design is for.

---

## Recommended workflow

## Step 1: prepare an agent folder

From the repo root:

```bash
python3 search_harness.py prepare-agent-folder \
  --config ncycle/my_cycle.json \
  --generation 0 \
  --repo-file stage3_3/base_train_gpt.py \
  --repo-file stage3_3/patches.py \
  --instructions "Build one executable survival pack on the fixed stage3_3 base. Use 4 to 6 distinct first-order candidates with stable controls and no compounds."
```

This writes:

```text
search_cycles/pgolf_family_search/gen_000/agent/
```

with:

- `AGENT_INSTRUCTIONS.md`
- `AGENT_PROMPT.txt`
- `request.json`
- `catalog.json`
- `postmortem_memory.json`
- `focus_files/`
- schema files

---

## Step 2: point an external agent at that folder

Recommended prompt:

```text
Read search_cycles/pgolf_family_search/gen_000/agent/AGENT_INSTRUCTIONS.md and complete the task fully.
```

The external agent runs in the **repo root** and has access to the **entire repo**.  
The `agent/` folder is only the task packet / contract.

The agent should create:

```text
search_cycles/pgolf_family_search/gen_000/
  compiled_generation.json
  track_spec.json
  verifier_report.json
  bundle/
```

The request packet now hard-gates survival packs to:

- one frozen base across controls and candidates
- aligned `DATA_PATH` / `TOKENIZER_PATH` / `VOCAB_SIZE`
- one candidate per `family_group`
- fewer candidates instead of mixed-base filler

---

## Step 3: check that the generation is staged

The generation is ready when you have:

```text
search_cycles/pgolf_family_search/gen_000/
  compiled_generation.json
  verifier_report.json
  bundle/
```

and inside:

```text
search_cycles/pgolf_family_search/gen_000/bundle/
  bundle_manifest.json
  search_harness.py
  slots/
```

Each slot should have:

```text
slots/<slot>/
  run.sh
  manifest.json
  <patched script>
  results/
```

---

## Step 4: run the tournament from the folder

The staged bundle is the portable execution unit.

From the repo root:

```bash
cd search_cycles/pgolf_family_search/gen_000/bundle
python3 search_harness.py execute-bundle --bundle bundle_manifest.json
```

This runs the tournament from **inside the generation folder**.

That is the cleanest execution path if you want to:

- prepare many generations first
- wait for GPU availability
- run one prepared generation later

---

## Step 5: extract results after execution

After the bundle run finishes, produce normalized metrics:

```bash
python3 search_harness_pgolf_extractor.py \
  --generation-dir search_cycles/pgolf_family_search/gen_000 \
  --output search_cycles/pgolf_family_search/gen_000/results_index.json
```

Then select survivors:

```bash
python3 search_harness.py select-survivors \
  --config ncycle/my_cycle.json \
  --generation 0
```

That will write:

```text
search_cycles/pgolf_family_search/gen_000/generation_summary.json
```

---

## What the bundle executor does

The bundle currently uses:

```text
bundle/search_harness.py
```

as the tournament runner.

Why:

1. the bundle can be rsynced and run without depending on the live repo state
2. the copied harness knows how to:
   - read `bundle_manifest.json`
   - assign GPUs
   - launch each slot
   - write `execution_report.json`

There is **not** currently a separate `tournament.sh`.

The tournament is driven by:

- `bundle/search_harness.py`
- `bundle/bundle_manifest.json`
- `slots/*/run.sh`

---

## Local execution command

If you want the repo-level command instead of entering the bundle directory:

```bash
python3 search_harness.py run-generation \
  --config ncycle/my_cycle.json \
  --generation 0 \
  --execute
```

But for queueing prepared generations and running later, the **bundle-local command** is usually clearer.

---

## Remote execution via SSH/rsync

The config supports a remote GPU host.

Current fields in `ncycle/my_cycle.json`:

```json
"remote": {
  "enabled": false,
  "ssh_target": "ubuntu@YOUR_GPU_HOST",
  "rsync_target": "ubuntu@YOUR_GPU_HOST:/data/parameter-golf/search_cycles/pgolf_family_search",
  "remote_python": "/data/pgolf_venv/bin/python",
  "pre_command": "cd /data/parameter-golf"
}
```

### What to fill in

- `ssh_target` — the host you SSH into
- `rsync_target` — where generation folders should be copied on that host
- `remote_python` — Python / venv on that host
- `pre_command` — usually `cd /data/parameter-golf`

### Example

```json
"remote": {
  "enabled": true,
  "ssh_target": "ubuntu@1.2.3.4",
  "rsync_target": "ubuntu@1.2.3.4:/data/parameter-golf/search_cycles/pgolf_family_search",
  "remote_python": "/data/pgolf_venv/bin/python",
  "pre_command": "cd /data/parameter-golf"
}
```

### Repo-level remote command

```bash
python3 search_harness.py run-generation \
  --config ncycle/my_cycle.json \
  --generation 0 \
  --execute --remote
```

That will:

1. rsync the prepared generation to the remote host
2. execute the staged bundle there
3. rsync results back
4. extract / select locally

### Manual remote command from the folder model

If you want to stay close to the clean-folder workflow:

1. rsync the entire generation folder yourself
2. SSH in
3. run inside the remote `bundle/` directory:

```bash
cd /data/parameter-golf/search_cycles/pgolf_family_search/gen_000/bundle
python3 search_harness.py execute-bundle --bundle bundle_manifest.json
```

4. pull results back
5. run extractor + selector locally

---

## Current metric extraction

The selector expects:

```text
metrics.score_bpb
```

The extractor gets that from `train.log`.

Current parser target:

- `final_int8_zlib_roundtrip_exact val_loss:... val_bpb:...`

If that line exists, it becomes:

```json
"metrics": {
  "score_bpb": ...,
  "post_quant_bpb": ...
}
```

If the final exact line is missing, the extractor falls back to the last validation step and records a note.

---

## What is currently working well

- repo-specific config exists
- generation folders are created under `search_cycles/pgolf_family_search/`
- agent packet includes:
  - family catalog
  - focus files
  - postmortem memory
- staged bundles are runnable from their own folder
- SSH/rsync config shape exists
- log extraction is wired for `train_gpt.py`-style logs

---

## Current limitation

The **direct `codex-generation` wrapper** may fail on some Codex CLI versions because the CLI is stricter about JSON schema requirements than the current wrapper expected.

So the **recommended practical path right now** is:

1. `prepare-agent-folder`
2. point Codex / Claude Code / Copilot at `agent/AGENT_INSTRUCTIONS.md`
3. run the staged `bundle/` later

This path is working and avoids the current schema mismatch in direct structured-output mode.

---

## Example end-to-end flow

### Prepare generation 0

```bash
python3 search_harness.py prepare-agent-folder \
  --config ncycle/my_cycle.json \
  --generation 0 \
  --repo-file train_gpt.py \
  --repo-file stage3_2/base_train_gpt.py \
  --repo-file stage3_5/base_train_gpt.py \
  --instructions "Build one executable first-order family-admission pack for this repo. Focus on state-conditioned control, checkpoint/export discipline, and stable controls. Use only existing executable patch families. Avoid compounds and lane mixing."
```

### Ask Codex manually

```text
Read search_cycles/pgolf_family_search/gen_000/agent/AGENT_INSTRUCTIONS.md and complete the task fully.
```

### Run later from the folder

```bash
cd search_cycles/pgolf_family_search/gen_000/bundle
python3 search_harness.py execute-bundle --bundle bundle_manifest.json
```

### Extract and rank

```bash
python3 search_harness_pgolf_extractor.py \
  --generation-dir search_cycles/pgolf_family_search/gen_000 \
  --output search_cycles/pgolf_family_search/gen_000/results_index.json

python3 search_harness.py select-survivors \
  --config ncycle/my_cycle.json \
  --generation 0
```

---

## Files worth checking when something looks wrong

- `search_cycles/pgolf_family_search/gen_000/compiled_generation.json`
- `search_cycles/pgolf_family_search/gen_000/verifier_report.json`
- `search_cycles/pgolf_family_search/gen_000/bundle/bundle_manifest.json`
- `search_cycles/pgolf_family_search/gen_000/bundle/slots/*/manifest.json`
- `search_cycles/pgolf_family_search/gen_000/bundle/slots/*/results/train.log`
- `search_cycles/pgolf_family_search/gen_000/results_index.json`
- `search_cycles/pgolf_family_search/gen_000/generation_summary.json`

---

## Short version

If you only remember one practical flow, use this:

1. prepare:

```bash
python3 search_harness.py prepare-agent-folder --config ncycle/my_cycle.json --generation 0 ...
```

2. point Codex at:

```text
search_cycles/pgolf_family_search/gen_000/agent/AGENT_INSTRUCTIONS.md
```

3. run later from:

```text
search_cycles/pgolf_family_search/gen_000/bundle/
```

with:

```bash
python3 search_harness.py execute-bundle --bundle bundle_manifest.json
```
