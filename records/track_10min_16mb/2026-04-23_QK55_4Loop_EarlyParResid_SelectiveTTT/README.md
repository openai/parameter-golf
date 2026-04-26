# val_bpb=1.1716 — 4-Loop Depth Recurrence + Early Parallel Residuals + Selective TTT

**Author:** Ismail Haddou (@ismailntl)
**Confirmed val_bpb:** 1.17158907 (1×H100, seed 1337)
**3-seed 8×H100 run:** pending compute credits

## Results

| step | val_bpb |
|---|---|
| 500 | 1.3673 |
| 1000 | 1.2682 |
| 1500 | 1.2188 |
| 2000 | 1.1675 |
| post-EMA | **1.17158907** |

Baseline (provided `train_gpt.py`) achieved 1.2977 at the same step count.

## What this submission does

Four changes applied together:

1. **QK-Gain 5.5** — attention key/query gain parameter tuned to 5.5
2. **NUM_LOOPS=3** — 4 recurrence passes through layers 3-5, giving 19 virtual layer executions from 11 physical layers
3. **Early Parallel Residuals** — GPT-J style parallel attention+MLP from layer 5 onward
4. **Selective TTT** — test-time training restricted to recurrent layers only, chunk size 24576, 4 epochs per chunk

## Architecture

11L × 512d × 8H / 4KV, MLP 4×, LeakyReLU(0.5)², Partial RoPE (16/64 dims), tied embeddings, logit softcap=30.0. SP1024 BPE tokenizer. Depth recurrence on layers 3-5 (activates at step 700, frac=0.35). Skip gates (sigmoid-gated U-Net connections). GPTQ SDClip int6/int8 + Brotli-11 compression.

## How to run

### Prerequisites

```bash
# Clone and enter repo
git clone https://github.com/ismailntl/parameter-golf.git
cd parameter-golf

# Download sp1024 dataset (if not already present)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

### Single run (1×H100, ablation)

```bash
RUN_ID=qk55_4loop \
DATA_DIR=./data \
VOCAB_SIZE=1024 \
ITERATIONS=2000 \
MAX_WALLCLOCK_SECONDS=0 \
SLIDING_WINDOW_ENABLED=1 \
TRAIN_BATCH_TOKENS=786432 \
torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-04-23_QK55_4Loop_EarlyParResid_SelectiveTTT/train_gpt.py
```

### Official 8×H100 submission run (10-min cap enforced)

```bash
RUN_ID=qk55_4loop_8gpu \
DATA_DIR=./data \
VOCAB_SIZE=1024 \
ITERATIONS=2000 \
MAX_WALLCLOCK_SECONDS=580 \
SLIDING_WINDOW_ENABLED=1 \
TRAIN_BATCH_TOKENS=786432 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-23_QK55_4Loop_EarlyParResid_SelectiveTTT/train_gpt.py
```

### Gauntlet runner (all experiments + ablations)

```bash
# Full gauntlet on 8×H100
bash gauntlet.sh --vocab 1024 --gpus 8 --incr-only

# Ablation on 1×H100
bash gauntlet.sh --vocab 1024 --gpus 1 --incr-only
```

`MAX_WALLCLOCK_SECONDS=580` is set automatically by the gauntlet, leaving 20s for GPTQ serialization within the 10-minute window.

### Additional experiments (in `experiments/`)

```bash
# DEQ Universal Transformer (1 physical block → fixed-point)
torchrun --standalone --nproc_per_node=8 experiments/train_gpt_deq.py

# Seed-LoRA (random linear map bases + stored LoRA adapters only)
LORA_RANK_ATTN=8 LORA_RANK_MLP=4 \
torchrun --standalone --nproc_per_node=8 experiments/train_gpt_seeds.py

# Mixture of Depths (50% token routing → ~2× more training steps)
MOD_CAPACITY=0.5 \
torchrun --standalone --nproc_per_node=8 experiments/train_gpt_mod.py
```

## Files

| File | Description |
|---|---|
| `train_gpt.py` | This submission's training script |
| `train_log_seed1337.log` | Full training log (seed 1337, 1×H100) |
| `train_log_baseline.log` | Baseline run log for comparison |
| `submission.json` | Metadata |
