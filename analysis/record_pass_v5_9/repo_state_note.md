# Repo State Note (V5.9)

## Scope Locked For This Pass
- This pass is execution-hardening only.
- Trunk remains `V5.2A`-style non-QAT.
- Single-hash remains mainline.
- Serializer autochoose remains mandatory.
- Packed remains optional/data-dependent, not default.
- Live promotion core remains `mlp_int5_to_int6`.
- QAT remains demoted for first-cycle paid spend.
- T0A/T0B/T0C are collapsed into one training family plus one post-train export sweep.

## Inputs Read Before Edits
- `analysis/record_pass_v5_4/final_report.md`
- `analysis/record_pass_v5_4/next_actions.md`
- `analysis/record_pass_v5_5/final_report.md`
- `analysis/record_pass_v5_5/launch_family_plan.md`
- `analysis/record_pass_v5_7/final_report.md`
- `analysis/record_pass_v5_7/consensus_export_decision.md`
- `analysis/record_pass_v5_7/next_actions.md`
- `analysis/record_pass_v5_8/decision_table.md` (used because `analysis/record_pass_v5_7/decision_table.md` is not present in repo)

## V5.9 Outcome Shape
- New branch: `codex/record-pass-v5_9-2026-03-23`
- New analysis folder: `analysis/record_pass_v5_9/`
- New record clone: `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/`
- Canonical launch packet files created in V5.9 record folder:
  - `runpod_preflight.py`
  - `launch_t0.sh`
  - `launch_export_sweep.sh`
  - `launch_t1_if_needed.sh`
  - `launch_donor_if_allowed.sh`
  - `launch_orchestrator.py`
  - `launch_config.json`

## What This Pass Explicitly Did Not Do
- No new architecture family search.
- No reopening donor-vs-trunk research for model selection.
- No new benchmark path added.
- No new paid training run executed.
