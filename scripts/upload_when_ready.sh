#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <src_root> <dest_root> [poll_seconds]" >&2
  exit 1
fi

src_root=$1
dest_root=$2
poll_seconds=${3:-120}
manifest_path="${src_root%/}/manifest.json"

if [[ "$poll_seconds" -le 0 ]]; then
  echo "poll_seconds must be positive" >&2
  exit 1
fi

while [[ ! -f "$manifest_path" ]]; do
  sleep "$poll_seconds"
done

bbb cptree "$src_root" "$dest_root"
