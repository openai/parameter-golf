#!/bin/bash
#SBATCH -p H100
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH --job-name=07c1_s64_qk5
#SBATCH --output=/netscratch/%u/parameter-golf/logs/%x-%j.out

set -euo pipefail
HOST_USER="${HOST_USER:-$USER}"
cd /netscratch/$HOST_USER/parameter-golf

srun --kill-on-bad-exit=1 \
  --container-image="/netscratch/$HOST_USER/containers/pytorch_25.02_fa3.sqsh" \
  --container-mounts="/netscratch/$HOST_USER/parameter-golf:/netscratch/$HOST_USER/parameter-golf,/fscratch/$HOST_USER:/fscratch/$HOST_USER" \
  --container-workdir="/netscratch/$HOST_USER/parameter-golf" \
  bash -lc '
export LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond,eth NCCL_P2P_LEVEL=NVL
export PIP_ROOT_USER_ACTION=ignore

python -c "import sentencepiece, brotli" 2>/dev/null || pip install --quiet sentencepiece brotli

DATA_PATH=/fscratch/ayach/parameter-golf-data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/fscratch/ayach/parameter-golf-data/tokenizers/fineweb_1024_bpe.model \
RUN_ID=07c1_pr1212_seed1337_s64_qkgain50 \
SEED=1337 \
MATRIX_LR=0.024 MATRIX_LR_LATE=0.019 \
SCALAR_LR=0.020 SCALAR_LR_LATE=0.038 \
TIED_EMBED_LR=0.022 MUON_MOMENTUM=0.985 WARMDOWN_ITERS=4000 \
TRAIN_BATCH_TOKENS=589824 MAX_WALLCLOCK_SECONDS=600 GPTQ_RESERVE_MS=0 \
NUM_LAYERS=12 BIGRAM_VOCAB_SIZE=5120 VE_DIM=128 \
WINDOW_SIZE=512 WINDOW_ATTN_LAYERS=2,4,6,8,10 \
SEQ_LENS_PER_GPU=2048,2048,2048,2048,2048,6144,6144,6144 \
SEQS_PER_GPU=36,36,36,36,36,10,10,10 \
TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=6144 EVAL_STRIDE=64 \
QK_GAIN_INIT=5.0 TTT_ENABLED=0 \
python -u records/track_non_record_16mb/2026-04-02_07c1_pr1212_ttt_evalfix/train_gpt.py
'
