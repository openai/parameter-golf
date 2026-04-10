#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
# Auto-discover trainer path (local or project root)
if [[ -f "${SCRIPT_DIR:-.}/train_gpt.py" ]]; then
    TRAINER_PATH="${SCRIPT_DIR:-.}/train_gpt.py"
elif [[ -f "$(cd "${SCRIPT_DIR:-.}/../../.." 2>/dev/null && pwd)/train_gpt.py" ]]; then
    TRAINER_PATH="$(cd "${SCRIPT_DIR:-.}/../../.." && pwd)/train_gpt.py"
else
    # Fallback for scripts that don't define SCRIPT_DIR
    TRAINER_PATH="./train_gpt.py"
fi
cd "$DIR"

echo "Syncing latest architectural fixes to winlaptop..."
scp train_gpt.py run_hybrid_win.bat winlaptop:C:/Users/Public/parameter-golf/records/track_10min_16mb/2026-03-25_Ternary_Feedback_TTT/

WIN_CHECK=$(ssh winlaptop 'powershell -Command "if (Get-Process python -ErrorAction SilentlyContinue) { Write-Output 1 } else { Write-Output 0 }"')

if [[ "$WIN_CHECK" == *"1"* ]]; then
    echo "Remote (CUDA) Windows benchmark is ALREADY RUNNING!"
else
    echo "Launching remote (CUDA) Windows benchmark in background via SSH..."
    ssh winlaptop 'cmd /c "cd C:\Users\Public\parameter-golf\records\track_10min_16mb\2026-03-25_Ternary_Feedback_TTT && run_hybrid_win.bat"' > win_ssh.out 2>&1 &
fi

if ! pgrep -f run_hybrid_mac.sh > /dev/null; then
    echo "Launching local (MLX) benchmark in background..."
    nohup bash run_hybrid_mac.sh > mac_nohup.out 2>&1 &
else
    echo "Local (MLX) Mac benchmark is ALREADY RUNNING!"
fi

echo ""
echo "=========================================================="
echo "Experiments Launched Successfully!"
echo "Run 'bash watch_experiments.sh' to view the Live Dashboard"
echo "=========================================================="
