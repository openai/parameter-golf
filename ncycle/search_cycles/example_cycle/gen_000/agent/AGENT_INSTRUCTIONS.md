# Agent Instructions

Work in the repository root. Do not modify unrelated source files.
Your task is to prepare generation `0` so `ncycle/search_cycles/example_cycle/gen_000/bundle` exists and is ready to run on the GPU setup later.

## Inputs to read first
- Request: `ncycle/search_cycles/example_cycle/gen_000/agent/request.json`
- Family catalog: `ncycle/search_cycles/example_cycle/gen_000/agent/catalog.json`
- Postmortem memory: `ncycle/search_cycles/example_cycle/gen_000/agent/postmortem_memory.json`
- Compiled schema: `search_harness_compiled_generation_schema.json`
- Verifier schema: `search_harness_verifier_report_schema.json`
- Compiled example: `search_harness_compiled_generation_example.json`
- Focus-file snapshots: `ncycle/search_cycles/example_cycle/gen_000/agent/focus_files`

## Focus repo files
- `train_gpt.py`
- `stage3_2/base_train_gpt.py`

## Required outputs to create
- `ncycle/search_cycles/example_cycle/gen_000/compiled_generation.json`
- `ncycle/search_cycles/example_cycle/gen_000/track_spec.json` (same content as compiled_generation.json is acceptable)
- `ncycle/search_cycles/example_cycle/gen_000/verifier_report.json`

## Hard rules
1. Use only executable families from the family catalog.
2. Keep controls unpatched.
3. Keep candidates first-order; do not create compounds.
4. Use postmortem memory from previous runs to avoid repeating mistakes.
5. If the user instructions ask for something impossible, choose the nearest executable pack and explain the gap in metadata plus verifier warnings.

## After writing the JSON files
Run: `python3 search_harness.py run-generation --config ncycle/my_cycle.json --generation 0 --dry-run`

## Success condition
- `ncycle/search_cycles/example_cycle/gen_000/bundle/bundle_manifest.json` exists
- slot folders exist under `ncycle/search_cycles/example_cycle/gen_000/bundle/slots/`
- the generation folder is ready for later remote GPU execution

## User instructions
Build one executable first-order family-admission pack focused on state-conditioned control.

Finish by leaving the generation folder populated and dry-run materialized.
