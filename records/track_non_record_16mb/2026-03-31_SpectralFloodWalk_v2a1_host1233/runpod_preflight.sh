#!/usr/bin/env bash
set -euo pipefail

RECORD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${SFW_RUNS_DIR:-${RECORD_DIR}/runs_host1233}"
mkdir -p "${RUNS_DIR}"

STAMP="${SFW_TIMESTAMP:-$(date -u +"%Y%m%dT%H%M%SZ")}"
OUT_PATH="${RUNS_DIR}/${STAMP}_preflight.log"
BENCHMARK_URL="${SFW_RUNPOD_BENCHMARK_URL:-https://raw.githubusercontent.com/NathanMaine/runpod-gpu-benchmark/main/pod-test.sh}"

echo "Writing preflight log to ${OUT_PATH}"
echo "Benchmark URL: ${BENCHMARK_URL}"
echo
echo "Heuristic keep band from Parameter Golf discussion #743:"
echo "  GEMM 4096x4096 bf16: < 0.50 ms good, > 0.70 ms reroll"
echo "  Memory bandwidth:    > 2800 GB/s good, < 2000 GB/s reroll"
echo "  Max GPU clock:       ~1980 MHz good, < 1800 MHz reroll"
echo

{
  echo "===== nvidia-smi ====="
  nvidia-smi || true
  echo
  echo "===== benchmark ====="
  curl -sL "${BENCHMARK_URL}" | bash
} | tee "${OUT_PATH}"

echo
echo "Saved preflight output to ${OUT_PATH}"
