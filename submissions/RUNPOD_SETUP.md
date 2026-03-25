# RunPod Setup Guide

## SSH Connection

```bash
# Use the RunPod-specific key (NOT the default id_ed25519)
ssh -i ~/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=no root@<IP> -p <PORT>

# SCP files
scp -i ~/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=no -P <PORT> <file> root@<IP>:/root/
```

## Current RunPod Instance (March 25, 2026)

- IP: `157.66.254.36`, Port: `14957`
- SSH: `ssh -i ~/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=no root@157.66.254.36 -p 14957`
- 8×H100 80GB HBM3 SXM
- Network volume at `/workspace` (persists between sessions)
- Repo moved to `/workspace/parameter-golf` with symlink from `/root/parameter-golf`
- **IMPORTANT: `/root` is ephemeral, `/workspace` persists. Keep everything in /workspace.**

## First-Time Setup

### 1. Install dependencies
```bash
pip install --break-system-packages brotli zstandard sentencepiece wandb huggingface-hub datasets tqdm
```

### 2. Clone repo
```bash
cd /root
git clone https://github.com/openai/parameter-golf.git
```

### 3. Download data (takes ~5 min)
```bash
cd /root/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
```

This creates:
- `/root/parameter-golf/data/datasets/fineweb10B_sp1024/` — 81 .bin files (80 train + 1 val)
- `/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`

### 4. Upload submission script
```bash
scp -i ~/.ssh/id_ed25519_runpod -o StrictHostKeyChecking=no -P 14957 \
  submissions/2026-03-25_14L_QEP_GPTQ_TTT/train_gpt.py \
  root@157.66.254.36:/root/parameter-golf/train_gpt_submission.py
```

## Running a Submission

### Single seed (seed 1337)
```bash
cd /root/parameter-golf

SEED=1337 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
NUM_LAYERS=14 BIGRAM_VOCAB_SIZE=8192 BIGRAM_DIM=64 \
MUON_WD=0.09 ADAM_WD=0.02 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 \
EVAL_STRIDE=76 MLP_ACTIVATION=leaky2 \
TTT_ENABLED=1 TTT_MODE=perwindow TTT_LR=0.002 TTT_EPOCHS=1 TTT_MOMENTUM=0.9 \
TTT_FREEZE_LAYERS=2 TTT_BATCH_SEQS=128 \
ROPE_BASE=50000 SWA_ENABLED=0 \
GPTQ_ENABLED=1 GPTQ_SAMPLES=256 QEP_ENABLED=1 \
WANDB_RUN_NAME=submission_seed1337 \
WANDB_PROJECT=parameter-golf \
WANDB_API_KEY=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 \
torchrun --standalone --nproc_per_node=8 train_gpt_submission.py
```

### 3-seed validation
Run the above with SEED=1337, SEED=42, SEED=7 separately.

### Background run with logging
```bash
nohup bash -c "<command above>" > submission_s1337.log 2>&1 &
```

## Expected Results

- Training: ~600s (5700 steps at ~105ms/step)
- QEP GPTQ: ~25s
- Compression: brotli-11
- Artifact: ~15.76MB (under 16MB ✓)
- Roundtrip BPP: ~1.1415
- TTT eval: ~551s at stride=76
- **Final BPP: ~1.1127**
- Total wall time: ~25 min

## Network Volume

Data is stored on network volume — should persist between pod restarts. Check:
```bash
ls /root/parameter-golf/data/datasets/fineweb10B_sp1024/*.bin | wc -l
# Should be 81
```

If data is missing, re-run the download command from step 3.

## Troubleshooting

- **wandb error**: Make sure `WANDB_API_KEY` is set
- **Port 29500 in use**: `pkill -9 -f torchrun; pkill -9 -f python; sleep 3` then retry
- **brotli not found**: `pip install --break-system-packages brotli`
- **flash-attn**: Should be pre-installed on RunPod PyTorch images. If not: `pip install --break-system-packages flash-attn`
