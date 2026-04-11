SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Load COMET_API_KEY from .env
[ -f "${SCRIPT_DIR}/../../.env" ] && source "${SCRIPT_DIR}/../../.env" || { [ -f .env ] && source .env; }
export COMET_API_KEY="${COMET_API_KEY:-}"
export PACK_BATCHES=1

bash "$SCRIPT_DIR/use_seq_packing.sh"
bash "$SCRIPT_DIR/without_seq_packing.sh"