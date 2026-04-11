set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load COMET_API_KEY from .env
[ -f "${SCRIPT_DIR}/../../.env" ] && source "${SCRIPT_DIR}/../../.env" || { [ -f .env ] && source .env; }
export COMET_API_KEY="${COMET_API_KEY:-}"

bash "$SCRIPT_DIR/1_048_576.sh"
bash "$SCRIPT_DIR/262_144.sh"
bash "$SCRIPT_DIR/131_072.sh"
bash "$SCRIPT_DIR/524_288.sh"
