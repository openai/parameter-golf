#!/bin/bash
# SHIVA NATRAJA — A2A Hub Runner
# φ² + φ⁻² = 3 · TRINITY · SHIVA · DANCE

set -e

cd "$(dirname "$0")"

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "[SHIVA] 🕉️ Shiva Natraja awakens... 🕉️"
echo "[SHIVA] Connecting four arms: NEON, RAILWAY, VIBEE, TRI-MCP-BROWSER"

# Install deps if needed
if [ ! -d node_modules ]; then
    echo "[SHIVA] Installing dependencies..."
    npm install
fi

# Start the hub
npm start
