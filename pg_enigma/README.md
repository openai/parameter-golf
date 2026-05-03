# pg_enigma

`pg_enigma` is the **campaign builder** that sits in front of `search_harness.py`.

It does **not** execute the GPU tournament itself. Its job is to turn a vague optimization brief into:

1. multiple independent hypothesis trajectories
2. an adversarial keep/drop decision
3. a multi-family campaign
4. several executable realizations per family
5. coherent lane-specific packs
6. staged downstream `search_harness.py` generations
7. family-level promotion and postmortem state for the next round

The key design choice is that the searchable object is a **campaign of families**, not one final patch list.

## Command index

All commands live under:

```bash
python3 pg_enigma/pg_enigma.py <command> ...
```

| Command | What it does | Main outputs | Resume |
| --- | --- | --- | --- |
| `init-config` | Write a reference config. | config JSON | No |
| `prepare-round` | Build `request.json` and snapshot focus/evidence files. | `request.json`, `focus_files/`, `evidence_files/` | No |
| `prepare-agent-folder` | Create an external-agent packet for a round. | `agent/` folder | No |
| `codex-round` | Run explore -> verify -> distill -> validate for one round. | `verification_report.json`, `campaign.json`, `family_dossiers/`, `compile_queue.json`, `SEARCH_HANDOFF.md`, `round_summary.json` | Yes |
| `validate-round` | Canonicalize and validate an already-written round. | same validated round artifacts as above | No |
| `debug-verifier` | Run the verifier prompt on real round artifacts via Copilot CLI and write debug artifacts. | `debug/<label>/prompt.txt`, Copilot raw output, normalized verification JSON, reuse summary | No |
| `runnable-families` | Print the current runnable family shortlist for a completed round. | stdout JSON, `runnable_families.json`, `family_status_report.json` | No |
| `run-round` | Run `codex-round`, `compile-families`, `build-pack-queue`, and `handoff-to-search-harness`. | validated round + family compiles + pack queue + staged handoffs | Yes |
| `compile-families` | Compile kept campaign families into several realization plans each. | `family_compiles/`, refreshed `pack_queue.json` | Yes |
| `build-pack-queue` | Group executable realizations into coherent packs. | `pack_queue.json` | No |
| `handoff-to-search-harness` | Stage selected packs as downstream `search_harness.py` generations. | `pack_handoffs.json`, per-pack derived configs and handoff prompts, downstream agent folders | Yes |
| `promote-families` | Read downstream pack evidence and promote families. | `promotion_report.json` | No |
| `postmortem-round` | Run the analyst stage after promotion/composition prep. | `postmortem_report.json` | Yes |
| `compose-survivors` | Build pairwise/hybrid placeholders from promoted solo families. | `composition_queue.json`, `hybrid_queue.json` | No |
| `advance-round` | Run postmortem for round N, then run round N+1 in the same cycle. | next round artifacts and staged handoffs | No |

## How one round works

One round is a directory under:

```text
pg_enigma/runs/<cycle_id>/round_000/
```

The lifecycle is:

1. **Prepare request**: merge config + user instructions + focus/evidence snapshots + history into `request.json`.
2. **Explore**: generate multiple independent trajectories, each with several candidate families.
3. **Verify**: score every candidate, keep only consequential ones, and reject/rewrite weak lines.
4. **Distill campaign**: convert the surviving candidates into one campaign with family metadata, pack order, promotion rules, and downstream instructions.
5. **Validate**: canonicalize the campaign and emit round-level derived artifacts like `family_dossiers/` and `compile_queue.json`.
6. **Compile families**: turn each kept family into several executable realizations of the same mechanism.
7. **Build pack queue**: group executable realizations into packs by `(pack_kind, lane, phase_window)`.
8. **Handoff to search_harness**: stage one downstream generation per selected pack, with exact slot lineage.
9. **Promote families**: after downstream execution, decide which families actually earned continued budget.
10. **Postmortem / compose / advance**: analyze evidence, queue pairwise composition work, and launch the next round.

## What `pg_enigma` writes

A mature round can contain:

```text
pg_enigma/runs/<cycle_id>/round_000/
  request.json
  focus_files/
  evidence_files/
  explorations/
    trajectory_001.json
    trajectory_002.json
    ...
  codex/
    attempt_001/
    attempt_002/
    summary.json
    compile_<family>/
    postmortem_001/
  verification_report.json
  campaign.json
  family_dossiers/
  compile_queue.json
  family_compiles/
  pack_queue.json
  pack_handoffs/
  pack_handoffs.json
  family_status_report.json
  runnable_families.json
  promotion_report.json
  composition_queue.json
  hybrid_queue.json
  postmortem_report.json
  SEARCH_HANDOFF.md
  SEARCH_HANDOFF_PROMPT.txt
  round_summary.json
```

Important boundaries:

- `pg_enigma` stops at **staged** downstream generations.
- `search_harness.py` is the layer that materializes bundles, executes slots, extracts results, and selects survivors.

## One-command use

Fresh round:

```bash
python3 pg_enigma/pg_enigma.py run-round \
  --config pg_enigma/reference_config.json \
  --round 0 \
  --instructions "Use run evidence to propose only consequential families for train_gpt.py." \
  --repo-file train_gpt.py \
  --evidence-file findings.md \
  --harness-config ncycle/my_cycle.json \
  --start-generation 0
```

This runs:

1. `codex-round`
2. `compile-families`
3. `build-pack-queue`
4. `handoff-to-search-harness`

It does **not** execute the downstream packs. It only stages them.

After `run-round`, the round folder also contains:

- `runnable_families.json` - the current shortlist you can run now
- `family_status_report.json` - the full split between runnable-now, blocked backlog, ready-for-pack, and already packed families

Resume a halted round without clearing prior work:

```bash
python3 pg_enigma/pg_enigma.py run-round \
  --config pg_enigma/reference_config.json \
  --round 0 \
  --resume \
  --harness-config ncycle/my_cycle.json \
  --start-generation 0
```

`--resume` is non-destructive:

- it reuses the stored `request.json`
- it reuses successful raw stage outputs already on disk
- it reuses valid `family_compiles/*.json`
- it keeps already staged pack handoffs instead of restaging them

When resuming, do **not** pass new `--repo-file` or `--evidence-file` values. If you pass `--instructions`, they must match the stored round request.

## Exact command behavior

### `init-config`

Writes a template config:

```bash
python3 pg_enigma/pg_enigma.py init-config --output pg_enigma/my_config.json
```

Use this when starting a new Enigma cycle definition.

### `prepare-round`

Creates the round request plus frozen snapshots of the files the round should reason over.

```bash
python3 pg_enigma/pg_enigma.py prepare-round \
  --config pg_enigma/reference_config.json \
  --round 0 \
  --instructions "..." \
  --repo-file train_gpt.py \
  --evidence-file findings.md
```

Reads:

- config
- instructions
- chosen focus/evidence files

Writes:

- `request.json`
- `focus_files/`
- `evidence_files/`

This is a pure preparation step. No model call happens here.

### `prepare-agent-folder`

Creates a repo-local packet for an external agent.

```bash
python3 pg_enigma/pg_enigma.py prepare-agent-folder \
  --config pg_enigma/reference_config.json \
  --round 0 \
  --instructions "..." \
  --repo-file train_gpt.py \
  --evidence-file findings.md
```

Writes an `agent/` folder containing the request, copied prompts/schemas, snapshots, and instructions for another agent to complete the round.

### `codex-round`

Runs the direct Enigma reasoning loop for one round:

1. explorer trajectories
2. verifier
3. campaign distiller
4. `validate-round`

```bash
python3 pg_enigma/pg_enigma.py codex-round \
  --config pg_enigma/reference_config.json \
  --round 0 \
  --instructions "..." \
  --repo-file train_gpt.py \
  --evidence-file findings.md
```

Resume:

```bash
python3 pg_enigma/pg_enigma.py codex-round \
  --config pg_enigma/reference_config.json \
  --round 0 \
  --resume
```

On success it leaves a validated round with:

- `verification_report.json`
- `campaign.json`
- `family_dossiers/`
- `compile_queue.json`
- `SEARCH_HANDOFF.md`
- `round_summary.json`

Fresh `codex-round` is destructive for mutable round-stage artifacts. `--resume` is not.

### `validate-round`

Canonicalizes and verifies an already written round:

```bash
python3 pg_enigma/pg_enigma.py validate-round \
  --config pg_enigma/reference_config.json \
  --round 0
```

This command expects:

- `request.json`
- all trajectory JSON files
- `verification_report.json`
- `campaign.json`

It then rewrites normalized forms and emits:

- `family_dossiers/`
- `compile_queue.json`
- `SEARCH_HANDOFF.md`
- `SEARCH_HANDOFF_PROMPT.txt`
- `round_summary.json`
- `family_status_report.json`
- `runnable_families.json`

### `debug-verifier`

Runs a real-world verifier debug pass through **Copilot CLI** instead of the normal `codex-round` automation.

Use this when you want to inspect:

- the exact verifier prompt
- whether the run took the **full** or **diff** review path
- the raw Copilot output
- the normalized verification JSON that the harness accepts

Example against a real stored attempt comparison:

```bash
python3 pg_enigma/pg_enigma.py debug-verifier \
  --config pg_enigma/pr1394_enigma_config.json \
  --round 0 \
  --attempt 2 \
  --reuse-from-attempt 1 \
  --mode auto \
  --model gpt-5.4 \
  --reasoning-effort xhigh \
  --label pr1394_realworld_diff
```

Mode behavior:

- `full` - review the whole slate
- `diff` - review only changed candidates against a prior attempt
- `auto` - use `diff` only when reuse is real; otherwise fall back to `full`

Artifacts are written under:

```text
pg_enigma/runs/<cycle_id>/round_000/debug/<label>/
```

including:

- `prompt.txt`
- `copilot_raw.txt`
- `copilot_extracted_json.txt` when Copilot emits narration before the final JSON
- `normalized_verification.json`
- `runnable_keep_families.json`
- `verification_reuse.json` when a prior attempt comparison was used
- `summary.json`

`summary.json` now also breaks kept verifier families into:

- `runnable_keep_ids`
- `blocked_keep_ids`

so the real-world debug pass tells you which keeps are executable now versus honest backlog.

### `runnable-families`

Print the current runnable shortlist for a round:

```bash
python3 pg_enigma/pg_enigma.py runnable-families \
  --config pg_enigma/reference_config.json \
  --round 0
```

This command refreshes `pack_queue.json` from the current round artifacts, rewrites:

- `family_status_report.json`
- `runnable_families.json`

and prints the JSON payload from `runnable_families.json` to stdout.

Selection behavior is intentionally practical:

- prefer families already staged inside packs
- otherwise fall back to families that are compile-ready for packs
- otherwise fall back to all `catalog_executable_now` campaign keeps

### `compile-families`

Compiles kept campaign families into multiple realization plans.

```bash
python3 pg_enigma/pg_enigma.py compile-families \
  --config pg_enigma/reference_config.json \
  --round 0
```

Optional narrowing:

```bash
python3 pg_enigma/pg_enigma.py compile-families \
  --config pg_enigma/reference_config.json \
  --round 0 \
  --family-id trajectory_003/trajectory_003_h3
```

Resume:

```bash
python3 pg_enigma/pg_enigma.py compile-families \
  --config pg_enigma/reference_config.json \
  --round 0 \
  --resume
```

Reads:

- `compile_queue.json`
- `family_dossiers/`
- `campaign.json`

Writes:

- `family_compiles/<candidate>.json`
- refreshed `pack_queue.json`
- refreshed `family_status_report.json`
- refreshed `runnable_families.json`

`--resume` skips already valid family compiles and reuses prior raw compiler attempts when available.

### `build-pack-queue`

Groups executable realizations into packs:

```bash
python3 pg_enigma/pg_enigma.py build-pack-queue \
  --config pg_enigma/reference_config.json \
  --round 0
```

This only includes families that are:

- `catalog_executable_now`
- in one of the initial executable pack kinds
- successfully compiled with verdict `READY`

Blocked families stay in `deferred_families`.

This command also refreshes:

- `family_status_report.json`
- `runnable_families.json`

Pack IDs are now unique and encode the lane:

```text
<pack_kind>_<lane>_<phase_window>_<nnn>
```

Example:

```text
selector_pack_selector_post_train_001
```

### `handoff-to-search-harness`

Stages selected packs as downstream `search_harness.py` generations.

All packs:

```bash
python3 pg_enigma/pg_enigma.py handoff-to-search-harness \
  --config pg_enigma/reference_config.json \
  --round 0 \
  --harness-config ncycle/my_cycle.json \
  --all-packs \
  --start-generation 0
```

One pack:

```bash
python3 pg_enigma/pg_enigma.py handoff-to-search-harness \
  --config pg_enigma/reference_config.json \
  --round 0 \
  --harness-config ncycle/my_cycle.json \
  --pack-id selector_pack_selector_post_train_001 \
  --start-generation 0
```

Resume:

```bash
python3 pg_enigma/pg_enigma.py handoff-to-search-harness \
  --config pg_enigma/reference_config.json \
  --round 0 \
  --harness-config ncycle/my_cycle.json \
  --all-packs \
  --start-generation 0 \
  --resume
```

Reads:

- `pack_queue.json`
- selected `family_dossiers/*.json`
- selected `family_compiles/*.json`

Writes:

- `pack_handoffs/<pack>_search_harness_config.json`
- `pack_handoffs/<pack>_SEARCH_HANDOFF.md`
- `pack_handoffs/<pack>_SEARCH_HANDOFF_PROMPT.txt`
- downstream `search_cycles/.../gen_xxx/agent/` folders
- `pack_handoffs.json`

`--resume` preserves existing `pack_id -> generation` assignments and stages only missing packs.

### `promote-families`

Reads executed downstream evidence and writes one family-level promotion report:

```bash
python3 pg_enigma/pg_enigma.py promote-families \
  --config pg_enigma/reference_config.json \
  --round 0
```

It expects downstream `search_harness.py` runs to have already produced:

- `generation_summary.json`
- `results_index.json`

for each staged pack generation in `pack_handoffs.json`.

Writes:

- `promotion_report.json`

Promotion is family-level, not slot-level.

### `postmortem-round`

Runs the analyst stage after pulling current family evidence together.

```bash
python3 pg_enigma/pg_enigma.py postmortem-round \
  --config pg_enigma/reference_config.json \
  --round 0
```

Resume:

```bash
python3 pg_enigma/pg_enigma.py postmortem-round \
  --config pg_enigma/reference_config.json \
  --round 0 \
  --resume
```

Important: this command automatically runs:

1. `promote-families`
2. `compose-survivors`
3. analyst postmortem

So the main output is `postmortem_report.json`, but it also refreshes promotion/composition state first.

### `compose-survivors`

Builds pairwise and hybrid placeholders from promoted solo families:

```bash
python3 pg_enigma/pg_enigma.py compose-survivors \
  --config pg_enigma/reference_config.json \
  --round 0
```

Writes:

- `composition_queue.json`
- `hybrid_queue.json`

This is queue-building only. It does not execute compositions.

### `advance-round`

Runs the next round in the same cycle after finishing postmortem on the previous round.

```bash
python3 pg_enigma/pg_enigma.py advance-round \
  --config pg_enigma/reference_config.json \
  --from-round 0 \
  --harness-config ncycle/my_cycle.json
```

What it actually does:

1. runs `postmortem-round` on `from_round`
2. chooses `to_round` (`from_round + 1` unless overridden)
3. inherits previous instructions/focus/evidence unless overridden
4. computes downstream start generation from prior `pack_handoffs.json` unless overridden
5. runs `run-round` for the next round

So `advance-round` is **not** just request staging. It launches the next round end-to-end through handoff.

## Resume rules

Supported resume commands are:

- `codex-round --resume`
- `run-round --resume`
- `compile-families --resume`
- `handoff-to-search-harness --resume`
- `postmortem-round --resume`

Behavior:

- Resume never clears the round.
- Stored artifacts are re-normalized before being trusted.
- If an old artifact is invalid under the current normalizer, resume fails loudly instead of silently accepting it.
- For `codex-round` / `run-round`, the stored round request is the source of truth.
- For `handoff-to-search-harness`, existing pack handoffs are kept and missing packs are appended.

## Downstream handoff boundary

After `pg_enigma` stages packs, the next layer is `search_harness.py`.

Typical downstream progression is:

1. execute the staged search-harness generation(s)
2. let `search_harness.py` produce `generation_summary.json` and `results_index.json`
3. return to `pg_enigma promote-families`
4. then `postmortem-round`
5. then `advance-round`

`search_harness.py` now also has a resumable preparation path:

```bash
python3 search_harness.py codex-generation \
  --config ncycle/my_cycle.json \
  --generation 0 \
  --resume
```

That matters because `pg_enigma` stages many downstream generations over time; resume support needs to exist on both sides of the handoff.

## Model configuration

`pg_enigma/reference_config.json` is set up for:

- model: `gpt-5.4`
- reasoning effort: `high`

via Codex CLI:

```text
model_reasoning_effort="high"
```

The config can also split models by stage:

- `explorer`
- `verifier`
- `distiller`
- `compiler`
- `analyst`

Using a different verifier model is useful when you want generation and critique to have different blind spots.

## What verifier and compiler should kill

The verifier and compiler are supposed to reject or rewrite:

- threshold nudges
- small scalar retunes
- child variants of unproven parents
- mixed-lane stories with no clear metric contract
- hidden compounds
- dishonest implementation claims
- fake pass@k where realizations are just renamed local tweaks

If an idea is real but too small, the verifier should push it **up a search level**.

If a family is real but the code proxies are weak, the compiler should emit **different executable realizations of the same mechanism**, not random mutation noise.
