#!/usr/bin/env bash
# sweep_runner.sh — orchestrate a batch of experiments from a TSV plan.
#
# TSV format (tab-separated, `#` = comment lines):
#   LABEL<TAB>KEY=VAL KEY=VAL ...
#
# Usage:
#   bash scripts/sweep_runner.sh scripts/sweeps/phase1_baseline.tsv
#   bash scripts/sweep_runner.sh all            # run all phase*.tsv in order
#
# Every run goes through run_experiment.sh, which never blocks the caller.

set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SWEEPS_DIR="scripts/sweeps"
RESULTS_CSV="${RESULTS_CSV:-logs/sweep/results.csv}"
STOP_REQUESTED=0

request_stop() {
  STOP_REQUESTED=1
  echo "[sweep] stop requested; will not launch more runs"
}

trap 'request_stop' INT TERM

run_one_tsv() {
  local tsv="$1"
  [[ -f "$tsv" ]] || { echo "[sweep] missing: $tsv"; return 1; }
  echo ""
  echo "############################################################"
  echo "# SWEEP FILE: $tsv"
  echo "############################################################"
  local total=0 done=0
  total=$(grep -cEv '^\s*(#|$)' "$tsv" || echo 0)
  while IFS=$'\t' read -r label overrides; do
    if [[ "$STOP_REQUESTED" == "1" ]]; then
      echo "[sweep] stopping before next row in $tsv"
      return 130
    fi
    # skip blank / comment
    [[ -z "$label" ]] && continue
    [[ "$label" =~ ^# ]] && continue
    done=$((done + 1))
    echo ""
    echo "--- [$done/$total] $label :: $overrides ---"
    # shellcheck disable=SC2086
    bash scripts/run_experiment.sh "$label" $overrides
    local rc=$?
    if [[ "$rc" == "130" || "$rc" == "143" ]]; then
      echo "[sweep] stop propagated from $label"
      return "$rc"
    fi
  done < "$tsv"
}

print_summary() {
  if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "[sweep] no results.csv yet"
    return
  fi
  echo ""
  echo "############################################################"
  echo "# SUMMARY — sorted by val_bpb ascending (blank bpb last)"
  echo "############################################################"
  # Print header + sorted rows (bpb col 5)
  ( head -1 "$RESULTS_CSV"; tail -n +2 "$RESULTS_CSV" \
      | awk -F, '{ key = ($5 == "" ? "9" : "0") $5; print key "\t" $0 }' \
      | sort | cut -f2- ) | column -t -s,
}

case "${1:-}" in
  ""|"-h"|"--help")
    echo "usage: sweep_runner.sh <plan.tsv | all>"
    echo ""
    echo "available plans:"
    ls -1 "$SWEEPS_DIR"/*.tsv 2>/dev/null || echo "  (none yet)"
    exit 1
    ;;
  "all")
    for tsv in "$SWEEPS_DIR"/phase*.tsv; do
      run_one_tsv "$tsv"
      rc=$?
      if [[ "$rc" == "130" || "$rc" == "143" ]]; then
        exit "$rc"
      fi
    done
    print_summary
    ;;
  "summary")
    print_summary
    ;;
  *)
    run_one_tsv "$1"
    rc=$?
    if [[ "$rc" == "130" || "$rc" == "143" ]]; then
      exit "$rc"
    fi
    print_summary
    ;;
esac
