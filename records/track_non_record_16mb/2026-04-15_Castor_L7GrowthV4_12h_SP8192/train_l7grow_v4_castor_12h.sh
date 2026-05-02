#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CASTOR_TRAIN_ENV="${CASTOR_TRAIN_ENV:-$ROOT/configs/train/l7grow_v4_castor_12h.env}"

exec "$ROOT/scripts/train_l7grow_v4_castor.sh"
