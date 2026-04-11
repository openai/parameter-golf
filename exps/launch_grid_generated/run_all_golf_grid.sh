set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load COMET_API_KEY from .env
[ -f "${SCRIPT_DIR}/../../.env" ] && source "${SCRIPT_DIR}/../../.env" || { [ -f .env ] && source .env; }
export COMET_API_KEY="${COMET_API_KEY:-}"
export PACK_BATCHES=1

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
