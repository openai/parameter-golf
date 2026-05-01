#!/usr/bin/env bash
set -euo pipefail

DOCS=/home/frosty40/parameter-golf-lab/data/docs_selected.jsonl
KEY=/home/frosty40/.ssh/id_ed25519_apollo
HOST=root@206.125.32.60
PORT=56335
KNOWN=/tmp/codex_vast_known_hosts

test -f "$DOCS"

ssh \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile="$KNOWN" \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=8 \
  -i "$KEY" \
  -p "$PORT" \
  "$HOST" \
  'cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x &&
   mkdir -p /workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers &&
   cp -f tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model /workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model &&
   /venv/main/bin/python3 stream_prepare_caseops_data.py \
     --docs - \
     --out /workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets \
     --sp /workspace/SOTA_FINAL/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
     --val-docs 50000 \
     --max-train-shards 80' \
  < "$DOCS"
