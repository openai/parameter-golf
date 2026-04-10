#!/bin/bash
set -euo pipefail

LOG_GLOB="${1:-logs/*}"
RUN_FILTER="${RUN_FILTER:-}"
tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

shopt -s nullglob
for f in ${LOG_GLOB}; do
  [[ -f "${f}" ]] || continue
  if [[ -n "${RUN_FILTER}" ]]; then
    if ! [[ "${f}" =~ ${RUN_FILTER} ]]; then
      continue
    fi
  fi

  speed="$(rg "step:[0-9]+/20000 train_loss:.*step_avg:" "${f}" 2>/dev/null | tail -n1 | sed -E 's/.*step_avg:([0-9.]+)ms.*/\1/' || true)"
  post="$(rg "DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:[0-9.]+" "${f}" 2>/dev/null | tail -n1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/' || true)"
  finish="$(rg "final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:[0-9.]+" "${f}" 2>/dev/null | tail -n1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/' || true)"
  roundtrip="$(rg "final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:[0-9.]+" "${f}" 2>/dev/null | tail -n1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/' || true)"

  if [[ -n "${speed}" && -n "${finish}" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\n" "${speed}" "${finish}" "${post:-NA}" "${roundtrip:-NA}" "${f}" >> "${tmp}"
  fi
done

if [[ ! -s "${tmp}" ]]; then
  echo "No matching logs with both speed and final sliding metrics."
  echo "Tip: set RUN_FILTER to narrow candidates, e.g. RUN_FILTER='medusa|SMK|nitrust'"
  exit 1
fi

echo "=== FASTEST (lowest step_avg ms) ==="
sort -n -k1,1 "${tmp}" | head -n 10 | awk -F'\t' '{printf "speed_ms=%s finish_bpb=%s post_ema_bpb=%s roundtrip_bpb=%s file=%s\n",$1,$2,$3,$4,$5}'

echo
echo "=== BEST FINISH (lowest final sliding bpb) ==="
sort -n -k2,2 "${tmp}" | head -n 10 | awk -F'\t' '{printf "finish_bpb=%s speed_ms=%s post_ema_bpb=%s roundtrip_bpb=%s file=%s\n",$2,$1,$3,$4,$5}'
