#!/usr/bin/env bash
set -euo pipefail

# Run from parameter-golf-rs/.

# Historical clean record-shaped timing run.
PG_WAIT=1 /tmp/pg-modal-venv312/bin/modal run deploy/run_detached.py \
  --modal-wait \
  --multi run \
  --spec /specs/frontier_1855_merged_target.toml \
  --mode record-shaped-proxy \
  --backend cuda-distributed \
  --artifact /output/frontier_v86_throughput_clean.pgrs \
  --result-json /output/frontier_v86_throughput_clean.json \
  --frontier-throughput-record-profile

# Current #2135-shaped speed-probe timing run. Use the direct function entrypoint
# so the job does not get lost in local-entrypoint spawn semantics.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_hybridst_budget_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_hybridst_pass1cache_v7.pgrs --result-json /output/frontier_2135_hybridst_pass1cache_timing_v7.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 4994 --force-cargo-clean"

# Best current <=120 ms systems profile that preserves two of three recurrent
# layer backward passes. Timing proof: 119.36 ms/step measured, 4,952 steps in
# 600 s; export proof: 15,640,347 strict decimal bytes.
/tmp/pg-modal-venv312/bin/modal run \
  deploy/run_detached.py::run_command_multi_string \
  --args "run --spec /specs/frontier_2135_hybridst1_budget_target.toml --mode record-shaped-proxy --backend cuda-distributed --artifact /output/frontier_2135_hybridst1_export_v1.pgrs --result-json /output/frontier_2135_hybridst1_export_v1.json --frontier-graph-record-profile --record-shaped-proxy-max-steps 4994 --export-record-shaped-artifact --force-cargo-clean"

# Proxy artifact BPB sanity only. This is intentionally not leaderboard-quality
# evidence because record-shaped proxy disables training loss; use a real record
# run for final BPB.
/tmp/pg-modal-venv312/bin/modal run deploy/run_detached.py \
  --modal-wait eval \
  --ttt-audit \
  --assert-score-no-mutation \
  --spec /specs/frontier_2135_hybridst1_budget_target.toml \
  --artifact /output/frontier_2135_hybridst1_export_v1.pgrs \
  --max-tokens 262144 \
  --eval-gpu-world-size 1 \
  --result-json /output/eval_frontier2135_hybridst1_262k_v1.json \
  --force-cargo-clean

# Regressed compact-u16 upload A/B.
PG_WAIT=1 /tmp/pg-modal-venv312/bin/modal run deploy/run_detached.py \
  --modal-wait \
  --multi run \
  --spec /specs/frontier_1855_merged_target.toml \
  --mode record-shaped-proxy \
  --backend cuda-distributed \
  --artifact /output/frontier_v87_u16_shift.pgrs \
  --result-json /output/frontier_v87_u16_shift.json \
  --frontier-throughput-record-profile \
  --enable-shifted-u16-batch-upload

# Artifact export proof command. This did not complete before submission due
# to Modal connectivity failure.
PG_WAIT=1 /tmp/pg-modal-venv312/bin/modal run deploy/run_detached.py \
  --modal-wait \
  --multi run \
  --spec /specs/frontier_1855_merged_target.toml \
  --mode record-shaped-proxy \
  --backend cuda-distributed \
  --artifact /output/frontier_v90_export_probe.pgrs \
  --result-json /output/frontier_v90_export_probe.json \
  --frontier-throughput-record-profile \
  --disable-shifted-u16-batch-upload \
  --export-record-shaped-artifact \
  --submission-code-bytes 1

# Local validation commands.
cargo check -q --features cuda -p pg-train -p pg-eval -p pg-data -p pg-kernels
cargo test -q -p pg-data
cargo test -q --features cuda -p pg-eval
cargo test -q --features cuda -p pg-train
python3 -m py_compile deploy/run_detached.py deploy/build_submission.py
