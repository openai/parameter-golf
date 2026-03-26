# Portability Patch Log (Behavior-Preserving)

## Code Patches
1. `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_orchestrator.py`
- Added repo-root detection fallback chain: `REPO_ROOT` env -> `git rev-parse --show-toplevel` -> ancestor search -> legacy fallback.
- Added optional `RECORD_DIR` env override.
- Kept launch modes, thresholds, and decision token logic unchanged.

2. `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/runpod_preflight.py`
- Added same repo-root fallback logic and `RECORD_DIR` override for token expansion.
- Kept check set, hard-failure policy, and output schema unchanged.

3. `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_t0.sh`
4. `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_export_sweep.sh`
5. `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_t1_if_needed.sh`
6. `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_donor_if_allowed.sh`
- Updated embedded `cfg_get` Python token-expansion helper to use the same robust repo-root detection chain.
- Added optional `RECORD_DIR` env override support.
- Kept commands, policies, budgets, thresholds, and branching behavior unchanged.

## Documentation Hygiene Patches
7. `analysis/record_pass_v5_9/final_report.md`
8. `analysis/record_pass_v5_9/decision_logic_audit.md`
9. `analysis/record_pass_v5_9/launch_graph_final.md`
10. `analysis/runpod_cycle_1/preflight_summary.md`
- Replaced machine-specific absolute path examples with repo-relative references.
- Did not alter factual status (preflight fail, pending paid run).

## Validation Performed
- Ran `--print-only` flows for `launch_t0.sh`, `launch_export_sweep.sh`, `launch_t1_if_needed.sh`, `launch_donor_if_allowed.sh`.
- Ran `launch_orchestrator.py --mode print_only`.
- Ran `runpod_preflight.py --dry-run`.
