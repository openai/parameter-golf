#!/usr/bin/env bash
#SBATCH --job-name=gdn_exp08_pure_softmax
#SBATCH --account=medcam
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/gdn_experiments/experiments/exp08_pure_softmax/logs/slurm_%j.out
#SBATCH --error=/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/gdn_experiments/experiments/exp08_pure_softmax/logs/slurm_%j.err
# ──────────────────────────────────────────────────────────────────────────────
# exp08_pure_softmax — Model H: Pure Sliding Window Attention (all SWA, no GDN)
# Control experiment for PR #1370 ablation study.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

EXPERIMENTS_DIR="/hpfs/scratch/gpfs/mcclec07/code/parameter_golf/gdn_hybrid/gdn_experiments/experiments"
EXPERIMENT_NAME="exp08_pure_softmax"
ARCH_MODE="H"

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Experiment: $EXPERIMENT_NAME"
echo "Arch mode: $ARCH_MODE"

cd "$EXPERIMENTS_DIR"
./run_experiment.sh "$EXPERIMENT_NAME" "$ARCH_MODE"

echo "Job $SLURM_JOB_ID finished at $(date)"
