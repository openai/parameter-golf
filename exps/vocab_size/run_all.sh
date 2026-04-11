set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load COMET_API_KEY from .env
[ -f "${SCRIPT_DIR}/../../.env" ] && source "${SCRIPT_DIR}/../../.env" || { [ -f .env ] && source .env; }
export COMET_API_KEY="${COMET_API_KEY:-}"

# bash "$SCRIPT_DIR/1024.sh"
bash "$SCRIPT_DIR/4096.sh"
bash "$SCRIPT_DIR/8192.sh"
bash "$SCRIPT_DIR/16384.sh"