#!/usr/bin/env bash
set -euo pipefail
# Авто-сгенерировано из check_model_size.ipynb
# Запуск по очереди: при ошибке в одном скрипте — остановка (set -e)
export COMET_API_KEY="wKvWIXBmWdm5O9w8buIWrqKEV"
export PACK_BATCHES=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== $(date -Iseconds) run_golf_L12_d256_mlp3_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L12_d256_mlp3_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L12_d256_mlp4_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L12_d256_mlp4_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L12_d256_mlp5_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L12_d256_mlp5_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L12_d320_mlp3_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L12_d320_mlp3_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L12_d320_mlp4_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L12_d320_mlp4_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L16_d256_mlp3_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L16_d256_mlp3_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L16_d256_mlp4_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L16_d256_mlp4_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L8_d256_mlp3_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L8_d256_mlp3_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L8_d256_mlp4_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L8_d256_mlp4_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L8_d256_mlp5_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L8_d256_mlp5_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L8_d320_mlp3_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L8_d320_mlp3_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L8_d320_mlp4_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L8_d320_mlp4_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L8_d320_mlp5_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L8_d320_mlp5_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L8_d384_mlp3_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L8_d384_mlp3_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L8_d384_mlp4_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L8_d384_mlp4_h8_kv4.sh"

echo "=== $(date -Iseconds) run_golf_L8_d448_mlp3_h8_kv4.sh ==="
bash "$SCRIPT_DIR/run_golf_L8_d448_mlp3_h8_kv4.sh"
