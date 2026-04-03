#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
MAX_PRICE="${MAX_PRICE:-999999}"

while true; do
  TS="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  RAW="$(vastai search offers "num_gpus=8 rentable=True" -t on-demand -o dph_total --raw 2>/dev/null || echo '[]')"

  MATCHES="$(echo "$RAW" | jq -c --argjson max "$MAX_PRICE" '
    map(select((.gpu_name // "") | test("H100"; "i")))
    | map(select((.dph_total // 1e9) <= $max))
    | sort_by(.dph_total)
  ')"

  COUNT="$(echo "$MATCHES" | jq 'length')"

  if [ "$COUNT" -gt 0 ]; then
    BEST="$(echo "$MATCHES" | jq -c '.[0]')"
    ASK_ID="$(echo "$BEST" | jq -r '(.ask_contract_id // .id // "?")')"
    PRICE="$(echo "$BEST" | jq -r '(.dph_total // "?")')"
    GPU_NAME="$(echo "$BEST" | jq -r '(.gpu_name // "?")')"
    REL="$(echo "$BEST" | jq -r '(.reliability2 // "?")')"
    echo "$TS FOUND_8xH100 count=$COUNT best_id=$ASK_ID price=$PRICE gpu='$GPU_NAME' reliability=$REL"
  else
    echo "$TS NO_8xH100"
  fi

  sleep "$INTERVAL_SECONDS"
done
