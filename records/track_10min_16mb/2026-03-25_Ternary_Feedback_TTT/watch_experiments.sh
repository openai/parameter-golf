#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "Booting Real-time TKA-H Monitoring Dashboard..."
python live_dashboard.py
