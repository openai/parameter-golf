#!/usr/bin/env bash
#SBATCH --job-name=gdn_exp11_kv_wider
#SBATCH --account=medcam
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/gdn_experiments/experiments/exp11_kv_share_wider/logs/slurm_%j.out
#SBATCH --error=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/gdn_experiments/experiments/exp11_kv_share_wider/logs/slurm_%j.err
# ──────────────────────────────────────────────────────────────────────────────
# exp11_kv_share_wider — 10L dim=544 GDN with KV sharing stride 2.
# Tests width redeployment: freed KV params → wider model (near iso-param).
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXPERIMENTS_DIR="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/gdn_experiments/experiments"
EXPERIMENT_NAME="exp11_kv_share_wider"
ARCH_MODE="K"

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Arch mode: $ARCH_MODE"

cd "$EXPERIMENTS_DIR"
./run_experiment.sh "$EXPERIMENT_NAME" "$ARCH_MODE"

echo "Job $SLURM_JOB_ID finished at $(date)"
