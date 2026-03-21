# Parameter Golf - End-to-End Process Guide

## Overview

This document captures the full process of setting up, training, and evaluating a Parameter Golf submission on RunPod with H100 GPUs. Use this as a repeatable playbook for future runs.

## Prerequisites

- RunPod account with API key (https://www.runpod.io/console/user/settings)
- `runpodctl` CLI installed
- `npx skills` (optional, for agent-assisted workflow)

## Step 1: Install runpodctl

```bash
# Linux
mkdir -p ~/.local/bin && curl -sL https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64.tar.gz | tar xz -C ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"

# macOS
brew install runpod/runpodctl/runpodctl

# Verify
runpodctl version
```

## Step 2: Configure API Key

```bash
export RUNPOD_API_KEY="your-key-here"
```

## Step 3: Set Up SSH Key

```bash
# Generate key if needed
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -q

# Add to RunPod (interactive)
runpodctl ssh add-key

# Or non-interactive:
echo -e "y\n$(cat ~/.ssh/id_ed25519.pub)" | runpodctl ssh add-key
```

> **Important**: Add the SSH key to your RunPod account BEFORE creating the pod. If the pod is created first, you must update the pod's `PUBLIC_KEY` env var and wait for it to restart:
> ```bash
> PUB_KEY=$(cat ~/.runpod/ssh/RunPod-Key-Go.pub)
> runpodctl pod update <pod-id> --env "{\"PUBLIC_KEY\":\"$PUB_KEY\",\"JUPYTER_PASSWORD\":\"parameter-golf\"}"
> ```
> The port number will change after restart — re-run `runpodctl ssh info <pod-id>` to get the new port.

## Step 4: Create Pod

```bash
# Check GPU availability
runpodctl gpu list

# Find template
runpodctl template search "parameter-golf"
# Template ID: y5cejece4j

# Create pod (single H100 SXM)
runpodctl pod create \
  --template-id y5cejece4j \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --name "parameter-golf-h100"

# For 8x H100 (competition target):
runpodctl pod create \
  --template-id y5cejece4j \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 8 \
  --name "parameter-golf-8xh100"
```

### Available H100 Variants
| GPU ID | Display Name | VRAM | Stock |
|--------|-------------|------|-------|
| `NVIDIA H100 80GB HBM3` | H100 SXM | 80 GB | High |
| `NVIDIA H100 NVL` | H100 NVL | 94 GB | Medium |
| `NVIDIA H100 PCIe` | H100 PCIe | 80 GB | High |

## Step 5: SSH into Pod

```bash
# Get SSH info (port may change after restarts)
runpodctl ssh info <pod-id>

# Connect
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port>
```

## Step 6: Clone Repo & Download Data

```bash
# On the pod:
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Install deps (if not in template image)
pip install --break-system-packages datasets sentencepiece huggingface-hub tqdm

# Download dataset (~sp1024 variant)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Verify
ls data/datasets/fineweb10B_sp1024/
ls data/tokenizers/
```

Expected files:
- `data/datasets/fineweb10B_sp1024/fineweb_train_*.bin` (80 shards)
- `data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`
- `data/tokenizers/fineweb_1024_bpe.model`
- `data/tokenizers/fineweb_1024_bpe.vocab`

## Step 7: Run Training

### Single GPU Baseline
```bash
RUN_ID=baseline_sp1024_try1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### 8x GPU (Competition Config)
```bash
RUN_ID=baseline_8gpu_try1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Key Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_ID` | random UUID | Run identifier, used in log filename |
| `DATA_PATH` | `./data/datasets/fineweb10B_sp1024` | Dataset directory |
| `TOKENIZER_PATH` | `./data/tokenizers/fineweb_1024_bpe.model` | Tokenizer path |
| `VOCAB_SIZE` | 1024 | Vocabulary size |
| `ITERATIONS` | 20000 | Max training steps |
| `MAX_WALLCLOCK_SECONDS` | 600 | Wall time cap (seconds). Set to 0 for unlimited |
| `TRAIN_BATCH_TOKENS` | 524288 | Tokens per training step |
| `VAL_LOSS_EVERY` | 1000 | Validate every N steps |
| `VAL_BATCH_SIZE` | 524288 | Validation batch tokens |
| `SEED` | 1337 | Random seed |

## Step 8: Collect Results

Training outputs are in `logs/<RUN_ID>.txt`. Key final metrics to collect:

```
final_int8_zlib_roundtrip val_loss:<val_loss> val_bpb:<val_bpb> eval_time:<ms>
final_int8_zlib_roundtrip_exact val_loss:<exact> val_bpb:<exact>
Serialized model int8+zlib: <bytes> bytes
Total submission size int8+zlib: <bytes> bytes
```

### What to Record Per Run
| Field | Source |
|-------|--------|
| try_id | `RUN_ID` env var |
| timestamp | Run start time (UTC) |
| gpu_config | GPU type x count |
| steps_completed | From `stopping_early` or last step line |
| wall_time_seconds | `train_time` from final step |
| val_loss | `final_int8_zlib_roundtrip_exact val_loss` |
| val_bpb | `final_int8_zlib_roundtrip_exact val_bpb` |
| compressed_size_bytes | `Total submission size int8+zlib` |
| under_16mb | Yes/No |
| stop_reason | `wallclock_cap` or `completed` |

## Step 9: Stop Pod

```bash
# Stop to save cost (keeps state for restart)
runpodctl pod stop <pod-id>

# Or delete entirely
runpodctl pod delete <pod-id>
```

## Lessons Learned (try1)

1. **Set up SSH keys before creating the pod.** Adding keys after creation requires a pod env update + restart, which changes the SSH port.

2. **Single H100 is wallclock-limited.** At ~427ms/step, only 1404/20000 steps complete in 600s. Use 8 GPUs for the competition config.

3. **The template image has most deps pre-installed.** Only `datasets`, `sentencepiece`, `huggingface-hub`, and `tqdm` may need manual installation.

4. **Environment variables don't persist across SSH sessions.** Always inline `RUNPOD_API_KEY` with commands or add it to shell profile.

5. **Pod ports change after restarts.** Always re-check with `runpodctl ssh info` after any pod restart/update.

6. **Baseline BPB (1.327) is far from SOTA (1.143).** Competitive runs need quantization-aware training, wider MLPs, and multi-GPU scaling.

## Cost Estimate

| Config | Cost/hr | ~10 min run cost |
|--------|---------|-----------------|
| 1x H100 SXM | $2.69 | ~$0.45 |
| 8x H100 SXM | ~$21.52 | ~$3.59 |

## Quick Reference Commands

```bash
# Check pod status
runpodctl pod get <pod-id>

# List all pods
runpodctl pod list

# Check account balance
runpodctl user

# View billing
runpodctl billing pods
```
