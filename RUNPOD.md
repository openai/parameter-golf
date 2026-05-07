# Parameter Golf — RunPod Workflow

## Fastest model on the leaderboard (April 2026)
**Score: 1.0810 bpb** — bigbag, *SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal Score-First TTT*  
File: `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py`

Key techniques:
- **SP8192** — 8192-token SentencePiece vocabulary
- **3-Layer Depth Recurrence** — layers 3-5 loop 3×, giving 17 effective layers from 11 physical
- **Parallel Residuals** (GPT-J style) — attention + MLP read same input from layer 7+
- **QK-Gain 5.25** — learnable per-head query scaling
- **Score-First TTT** — SGD test-time training on eval chunks, scores first then updates
- **GPTQ SDClip int6/int8** + Brotli-11 compression to stay under 16 MB

---

## 1. Launch a RunPod pod

1. Go to [runpod.io](https://www.runpod.io) → **GPU Cloud** → **New Pod**
2. Use the official template: [Parameter Golf Template](https://www.runpod.io/console/gpu-cloud)
   - Or choose any pod with a CUDA image (e.g., `runpod/pytorch:2.2.0-py3.10-cuda12.1-devel`)
3. Enable **SSH terminal access**
4. For experiments: **1×H100** (~$3/hr). For leaderboard: **8×H100 SXM** (~$20/hr)

## 2. SSH into the pod

```bash
ssh root@<pod-ip> -p <pod-port>
```

## 3. Run setup on the pod

```bash
curl -fsSL https://raw.githubusercontent.com/openai/parameter-golf/main/... \
  | bash
# OR after syncing this repo:
bash runpod_setup.sh
```

## 4. Sync your local changes to the pod

Set your pod connection info:
```bash
export RUNPOD_HOST="root@213.34.xx.xx"
export RUNPOD_PORT="22204"
```

One-shot sync:
```bash
bash sync_to_runpod.sh
```

Watch mode (auto-sync on file save):
```bash
bash sync_to_runpod.sh --watch
```

## 5. Training commands

### Quick test (1×H100, sp1024 baseline, ~10 min)
```bash
cd /workspace/parameter-golf
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### Reproduce current SOTA (8×H100 SXM, sp8192)
```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT
RUN_ID=sota_repro \
DATA_PATH=../../../data/datasets/fineweb10B_sp8192/ \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_8192_spm.model \
VOCAB_SIZE=8192 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Unlimited compute (no 10-min cap)
```bash
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=200 \
RUN_ID=my_long_run \
... torchrun ...
```

## 6. Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `RUN_ID` | required | Name for this run's output |
| `DATA_PATH` | required | Path to FineWeb dataset shards |
| `TOKENIZER_PATH` | required | Path to .model tokenizer |
| `VOCAB_SIZE` | `1024` | Must match tokenizer (1024 or 8192) |
| `MAX_WALLCLOCK_SECONDS` | `600` | Set to `0` for unlimited |
| `VAL_LOSS_EVERY` | `0` | Print val loss every N steps |
| `VAL_BATCH_SIZE` | `8192` | Tokens per val batch |
| `ITERATIONS` | auto | Override training step count |
