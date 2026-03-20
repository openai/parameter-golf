#!/bin/bash
set -euo pipefail

python3 -m py_compile modal_app.py train_gpt.py

SWEEP_ID="autoresearch_$(date +%Y%m%d_%H%M%S)_$$"
SUMMARY_JSON="$(mktemp)"
export SWEEP_ID

uv run --with modal --with huggingface-hub python - <<'PY' > "$SUMMARY_JSON"
import json
import os
import modal_app

with modal_app.app.run():
    result = modal_app.sweep_jobs.remote(
        variant="sp1024",
        train_shards=1,
        gpu_count=1,
        sweep_id=os.environ["SWEEP_ID"],
        iterations=300,
        max_wallclock_seconds=180,
        val_loss_every=0,
        train_log_every=100,
        base_env_overrides="",
        num_layers_values="9,12",
        model_dim_values="512,640",
        num_heads_values="8",
        num_kv_heads_values="4",
        mlp_mult_values="2",
    )
print(json.dumps(result))
PY

python3 - <<'PY' "$SUMMARY_JSON"
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
best = summary.get("best") or {}
size = best.get("submission_size_bytes")
constraint_ok = 1 if isinstance(size, int) and size < 16_000_000 else 0

print(f"METRIC best_val_bpb={best.get('val_bpb', 0)}")
print(f"METRIC best_submission_size_bytes={size or 0}")
print(f"METRIC best_compressed_model_bytes={best.get('compressed_model_bytes') or 0}")
print(f"METRIC best_model_params={best.get('model_params') or 0}")
print(f"METRIC constraint_ok={constraint_ok}")
print(f"METRIC job_count={summary.get('job_count', 0)}")
PY
