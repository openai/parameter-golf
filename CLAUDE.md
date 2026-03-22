# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parameter Golf is OpenAI's Model Craft Challenge: train the best language model that fits in a **16MB artifact** (code + compressed weights) in under **10 minutes on 8×H100s**, optimized for bits-per-byte (BPB) on FineWeb validation.

## Commands

### Training (multi-GPU)
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Training (single GPU)
```bash
python train_gpt.py
```

### Download data
```bash
python data/cached_challenge_fineweb.py
```

All model hyperparameters are configured via environment variables (see `Hyperparameters` dataclass in train_gpt.py). Key ones:
- `DATA_PATH`, `TOKENIZER_PATH` — dataset/tokenizer locations
- `VOCAB_SIZE`, `NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `MLP_MULT` — architecture
- `ITERATIONS`, `MAX_WALLCLOCK_SECONDS`, `TRAIN_BATCH_TOKENS`, `TRAIN_SEQ_LEN` — training budget
- `MATRIX_LR`, `SCALAR_LR`, `EMBED_LR`, `TIED_EMBED_LR`, `HEAD_LR` — per-group learning rates

There is no build system, test suite, or linter. The project is a single training script.

## Architecture

### train_gpt.py (~1100 lines, single-file constraint)

The entire model, training loop, data loading, evaluation, and serialization live in one file. The challenge rules require all code in `train_gpt.py` (hard limit: 1500 lines).

**Model (GPT class):** Transformer with RMSNorm, RoPE, Grouped Query Attention (GQA), ReLU² MLP, tied embeddings, logit softcapping, and skip connections between layers.

**Optimizer:** Muon (Newton-Schulz orthogonalization) for 2D matrix parameters; Adam for embeddings and scalar/control parameters. Separate learning rate groups for embeddings, matrices, scalars, and optional untied head.

**Data pipeline:** Binary shards (256-int header + uint16 tokens) → `TokenStream` → `DistributedTokenLoader` → sequential streaming batches. No random sampling.

**Evaluation:** Tokenizer-agnostic BPB metric computed via SentencePiece byte-accounting lookup tables, handling token boundaries and leading spaces correctly.

**Serialization:** Int8 post-training quantization with per-row scales for 2D tensors, FP16 passthrough for small tensors, zlib compression (level 9). Final artifact must be ≤16,000,000 bytes.

### train_gpt_mlx.py

MLX port for Apple Silicon development. Same architecture, different backend.

## Challenge Rules (key constraints)

- Artifact = `len(open("train_gpt.py").read().encode()) + len(compressed_model_bytes)` ≤ 16MB
- Training must complete in ≤10 minutes wallclock on 8×H100s
- Cannot access validation data during training (test-time training on already-evaluated tokens is allowed)
- New SOTA requires ≥0.005 nats BPB improvement with p < 0.01 statistical significance
- Default config: 1024 vocab (SentencePiece BPE), 9 layers, 512 dim, 8 heads, 4 KV heads

## Records

Submissions live in `records/track_10min_16mb/` with each containing a `train_gpt.py`, `submission.json` (val_bpb, bytes_total, author), `train.log`, and `README.md` describing techniques used.

## RunPod

Use `$RUNPOD_API_KEY` with `runpodctl`. SSH key: `/home/work/.ssh/id_ed25519`.

### Create H100 pod (parameter-golf template)
```bash
PUB_KEY=$(cat /home/work/.ssh/id_ed25519.pub)
$RUNPOD_API_KEY runpodctl pod create \
  --template-id y5cejece4j \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 1 \
  --name "param-golf" \
  --volume-in-gb 50 --container-disk-in-gb 50 \
  --ports "8888/http,22/tcp" --ssh \
  --env "{\"JUPYTER_PASSWORD\":\"parameter-golf\",\"PUBLIC_KEY\":\"$PUB_KEY\"}"
```

### SSH into pod
```bash
ssh -i /home/work/.ssh/id_ed25519 root@<IP> -p <PORT>
```

### List / stop / delete pods
```bash
$RUNPOD_API_KEY runpodctl pod list
$RUNPOD_API_KEY runpodctl pod stop <POD_ID>
$RUNPOD_API_KEY runpodctl pod delete <POD_ID>
```

### Create spot (interruptible) H100 — $1.75/hr vs $2.69 on-demand
```bash
PUB_KEY=$(cat /home/work/.ssh/id_ed25519.pub)
curl -s -X POST https://api.runpod.io/graphql \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -d "{\"query\": \"mutation { podRentInterruptable(input: { name: \\\"param-golf-spot\\\", templateId: \\\"y5cejece4j\\\", gpuTypeId: \\\"NVIDIA H100 80GB HBM3\\\", gpuCount: 1, volumeInGb: 50, containerDiskInGb: 50, cloudType: SECURE, startSsh: true, ports: \\\"8888/http,22/tcp\\\", bidPerGpu: 1.75, env: [{key: \\\"JUPYTER_PASSWORD\\\", value: \\\"parameter-golf\\\"}, {key: \\\"PUBLIC_KEY\\\", value: \\\"$PUB_KEY\\\"}] }) { id costPerHr desiredStatus machine { gpuDisplayName location } } }\"}"
```

### Key info
- Template ID: `y5cejece4j` (runpod/parameter-golf:latest)
- H100 SXM GPU ID: `NVIDIA H100 80GB HBM3` (on-demand ~$2.69/hr, spot ~$1.75/hr)
- Image has Python 3.12, PyTorch 2.9.1, all deps pre-installed
- Data download: `python3 data/cached_challenge_fineweb.py --variant sp1024` (run on pod)
- Template doesn't auto-clone — run `git clone https://github.com/openai/parameter-golf.git` on pod
- Need `pip install --break-system-packages zstandard` on the pod

### Deployment script (`run_on_runpod.sh`)
```bash
./run_on_runpod.sh              # Create spot pod, setup, train
./run_on_runpod.sh --status     # Pod status + SSH command
./run_on_runpod.sh --logs       # Tail training logs
./run_on_runpod.sh --results    # Show key metrics
./run_on_runpod.sh --save-log <tag>  # Save full log to logs/<timestamp>_<tag>.log
./run_on_runpod.sh --stop       # Stop pod
./run_on_runpod.sh --delete     # Delete pod
```

Override GPU config with env vars:
```bash
GPU_COUNT=8 BID_PRICE=1.75 ./run_on_runpod.sh           # 8xH100 spot ($14/hr)
GPU_COUNT=1 BID_PRICE=1.75 ./run_on_runpod.sh           # 1xH100 spot ($1.75/hr)
GPU_ID="NVIDIA RTX PRO 4500 Blackwell" BID_PRICE=0.27 ./run_on_runpod.sh  # cheap size test
TRAIN_SHARDS=1 ./run_on_runpod.sh                       # minimal data for quick iteration
```

### Logging
Save every training run's log after completion:
```bash
./run_on_runpod.sh --save-log "11L_VR1_GA1_prune3pct"
```
This saves to `logs/<timestamp>_<tag>.log` and `logs/<timestamp>_<tag>.summary` with key metrics extracted.
Logs track: config, val_bpb, artifact size, steps, timing — essential for comparing experiments.

### Cost-saving tips
- **Always delete pods after saving logs/results** — run `./run_on_runpod.sh --save-log <tag>` then `./run_on_runpod.sh --delete`
- **Test artifact size on cheap GPUs first** — use RTX PRO 4500 spot ($0.27/hr) to verify model fits 16MB before running on H100
- **Use `TRAIN_SHARDS=1`** for quick iteration (less data download time)
- **Use `EVAL_STRIDE=0`** to skip slow sliding window eval on single GPU (takes hours; only useful on 8xH100)
- **Use `EMA_ENABLED=0`** on single GPU — EMA kills throughput (~32% slower) but works fine on multi-GPU
- **Always `--stop` or `--delete` pods when done** — spot 8xH100 is $14/hr
- **Spot instances get preempted** — always use `nohup` and check if pod is still alive before reading logs
- **Shallow clone** (`git clone --depth 1`) saves ~30s on pod setup
- **TTT (test-time training) needs H100** — OOMs on 32GB GPUs (RTX 4500). Only enable on H100+
- **TTT on single GPU is very slow** — 20 epochs takes hours on 1xH100 but ~5 min on 8xH100
