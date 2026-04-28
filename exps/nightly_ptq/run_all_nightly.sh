#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# НАСТРОЙКИ ДЛЯ БЫСТРОЙ ВАЛИДАЦИИ
# Раскомментируй эти строки, чтобы прогнать скрипты на 10 итерациях и проверить,
# что ничего не падает (OOM, ошибки компиляции и т.д.)
# ==============================================================================
# export NIGHTLY_ITERATIONS=10
# export NIGHTLY_WALLCLOCK=30
# export NIGHTLY_COMET_KEY="your_api_key_here"

echo "Starting 60 nightly experiments..."
for script in "$SCRIPT_DIR"/exp_*.sh; do
    echo "========================================================="
    echo "Running $script..."
    echo "========================================================="
    bash "$script"
done
echo "All done!"
