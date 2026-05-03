#!/usr/bin/env bash
set -euo pipefail

# Run from parameter-golf-rs/.

# Current best clean record-shaped timing run.
PG_WAIT=1 /tmp/pg-modal-venv312/bin/modal run deploy/run_detached.py \
  --modal-wait \
  --multi run \
  --spec /specs/frontier_1855_merged_target.toml \
  --mode record-shaped-proxy \
  --backend cuda-distributed \
  --artifact /output/frontier_v86_throughput_clean.pgrs \
  --result-json /output/frontier_v86_throughput_clean.json \
  --frontier-throughput-record-profile

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

