#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Optional filters. Leave empty to disable.
MEDUSA_MODEL_PARAMS="${MEDUSA_MODEL_PARAMS:-14487088}"
MEDUSA_REQUIRE_PATTERNS="${MEDUSA_REQUIRE_PATTERNS:-}"
TAG="${TAG:-fastest_medusa_sota_$(date +%Y%m%d_%H%M%S)}"

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

shopt -s nullglob
for f in logs/*.txt logs/*.log; do
  [[ -f "${f}" ]] || continue

  if [[ -n "${MEDUSA_MODEL_PARAMS}" ]]; then
    if ! rg -q "model_params:${MEDUSA_MODEL_PARAMS}" "${f}"; then
      continue
    fi
  fi

  if [[ -n "${MEDUSA_REQUIRE_PATTERNS}" ]]; then
    ok=1
    IFS=',' read -r -a pats <<< "${MEDUSA_REQUIRE_PATTERNS}"
    for p in "${pats[@]}"; do
      [[ -z "${p}" ]] && continue
      if ! rg -q "${p}" "${f}"; then
        ok=0
        break
      fi
    done
    [[ "${ok}" -eq 1 ]] || continue
  fi

  speed="$(rg "step:[0-9]+/20000 train_loss:.*step_avg:" "${f}" 2>/dev/null | tail -n1 | sed -E 's/.*step_avg:([0-9.]+)ms.*/\1/' || true)"
  post_ema="$(rg "DIAGNOSTIC post_ema val_loss:[0-9.]+ val_bpb:[0-9.]+" "${f}" 2>/dev/null | tail -n1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/' || true)"
  roundtrip="$(rg "final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:[0-9.]+" "${f}" 2>/dev/null | tail -n1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/' || true)"
  sliding="$(rg "final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:[0-9.]+" "${f}" 2>/dev/null | tail -n1 | sed -E 's/.*val_bpb:([0-9.]+).*/\1/' || true)"

  [[ -n "${speed}" ]] || continue
  [[ -n "${sliding}" ]] || sliding="9999"
  [[ -n "${post_ema}" ]] || post_ema="NA"
  [[ -n "${roundtrip}" ]] || roundtrip="NA"
  printf "%s\t%s\t%s\t%s\t%s\n" "${speed}" "${sliding}" "${post_ema}" "${roundtrip}" "${f}" >> "${tmp}"
done

if [[ ! -s "${tmp}" ]]; then
  echo "No matching Medusa logs found."
  echo "Try loosening filters, e.g.:"
  echo "  MEDUSA_MODEL_PARAMS='' MEDUSA_REQUIRE_PATTERNS='' bash experiments/Medusa_II/promote_fastest_sota.sh"
  exit 1
fi

best="$(sort -n -k1,1 -k2,2 "${tmp}" | head -n1)"
best_speed="$(echo "${best}" | awk -F'\t' '{print $1}')"
best_sliding="$(echo "${best}" | awk -F'\t' '{print $2}')"
best_post="$(echo "${best}" | awk -F'\t' '{print $3}')"
best_roundtrip="$(echo "${best}" | awk -F'\t' '{print $4}')"
best_log="$(echo "${best}" | awk -F'\t' '{print $5}')"

sota_dir="${SCRIPT_DIR}/sota/${TAG}"
safe_dir="${SCRIPT_DIR}/endgame_safe/${TAG}"
mkdir -p "${sota_dir}" "${safe_dir}"

cp "${best_log}" "${sota_dir}/source.log"
cp "${SCRIPT_DIR}/run.sh" "${sota_dir}/run.sota.sh"
cp "${SCRIPT_DIR}/train_gpt.py" "${sota_dir}/train_gpt.sota.py"
cp "${SCRIPT_DIR}/run.sh" "${safe_dir}/run.sh"
cp "${SCRIPT_DIR}/train_gpt.py" "${safe_dir}/train_gpt.py"

# Best effort extraction of in-log train_gpt source if present at top of log.
sep_line="$(rg -n '^={40,}$' "${best_log}" | head -n1 | cut -d: -f1 || true)"
if [[ -n "${sep_line}" && "${sep_line}" -gt 1 ]]; then
  head -n "$((sep_line - 1))" "${best_log}" > "${sota_dir}/train_gpt.from_log.py" || true
fi

cat > "${sota_dir}/SUMMARY.txt" <<EOF
tag: ${TAG}
source_log: ${best_log}
speed_step_avg_ms: ${best_speed}
post_ema_val_bpb: ${best_post}
final_int6_roundtrip_exact_val_bpb: ${best_roundtrip}
final_int6_sliding_window_exact_val_bpb: ${best_sliding}
model_params_filter: ${MEDUSA_MODEL_PARAMS}
required_patterns_filter: ${MEDUSA_REQUIRE_PATTERNS}
promoted_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
git_commit: $(git rev-parse --short HEAD)
EOF

ln -sfn "${TAG}" "${SCRIPT_DIR}/sota/CURRENT"
ln -sfn "${TAG}" "${SCRIPT_DIR}/endgame_safe/CURRENT"

echo "Promoted fastest Medusa run to SOTA."
echo "  source_log: ${best_log}"
echo "  speed_step_avg_ms: ${best_speed}"
echo "  final_int6_sliding_window_exact_val_bpb: ${best_sliding}"
echo "  SOTA snapshot: ${sota_dir}"
echo "  Endgame safe copy: ${safe_dir}"
