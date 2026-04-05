#!/usr/bin/env bash
#SBATCH --job-name=gdn_exp09_kv_share
#SBATCH --account=medcam
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/gdn_experiments/experiments/exp09_kv_share/logs/slurm_%j.out
#SBATCH --error=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/gdn_experiments/experiments/exp09_kv_share/logs/slurm_%j.err
# ──────────────────────────────────────────────────────────────────────────────
# exp09_kv_share — Pure GDN 10L dim=512 with KV sharing stride 2.
# Tests the pure effect of adjacent-layer KV sharing (saves ~2.64M params).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXPERIMENTS_DIR="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/gdn_experiments/experiments"
EXPERIMENT_NAME="exp09_kv_share"
ARCH_MODE="I"

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Arch mode: $ARCH_MODE"

cd "$EXPERIMENTS_DIR"
./run_experiment.sh "$EXPERIMENT_NAME" "$ARCH_MODE"

echo "Job $SLURM_JOB_ID finished at $(date)"
