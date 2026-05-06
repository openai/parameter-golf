#!/bin/bash
#SBATCH --job-name=pgolf-ttt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/mahyavanshi.r/parameter-golf/logs/pgolf_%j.out
#SBATCH --error=/scratch/mahyavanshi.r/parameter-golf/logs/pgolf_%j.err

source /scratch/mahyavanshi.r/envs/pgolf/bin/activate

cd /scratch/mahyavanshi.r/parameter-golf

mkdir -p logs

SEED=${SEED:-42}

echo "=== Parameter Golf: SOTA + LoRA TTT ==="
echo "=== Seed: $SEED | Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== Start: $(date) ==="
nvidia-smi

# No wallclock cap for dev runs on 1 GPU
MAX_WALLCLOCK_SECONDS=0 \
SEED=$SEED \
RUN_ID="sota_ttt_seed${SEED}" \
TTT_ENABLED=1 \
TTT_LORA_RANK=${TTT_LORA_RANK:-8} \
TTT_LORA_LR=${TTT_LORA_LR:-0.01} \
TTT_CHUNK_SIZE=${TTT_CHUNK_SIZE:-256} \
TTT_BATCH_SIZE=${TTT_BATCH_SIZE:-64} \
TTT_GRAD_STEPS=${TTT_GRAD_STEPS:-1} \
torchrun --standalone --nproc_per_node=1 \
    records/track_10min_16mb/2026-03-22_SOTA_LoRA_TTT/train_gpt.py

echo "=== Done: $(date) ==="
