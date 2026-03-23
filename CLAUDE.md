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
- `TTT_ENABLED`, `TTT_OPTIMIZER` (adamw/muon/sgd), `TTT_EPOCHS`, `TTT_LR`, `TTT_COSINE` — test-time training
- `LEAKY_SLOPE` (0.0=ReLU², 0.5=LeakyReLU(0.5)²), `GPTQ_ENABLED` — activation & quantization
- `EMA_ENABLED`, `SWA_ENABLED`, `LATE_QAT`, `VALUE_RESIDUAL`, `GATED_ATTENTION`, `XSA_LAST_N`, `LN_SCALE`

There is no build system, test suite, or linter. The project is a single training script.

## Architecture

### train_gpt.py (~1487 lines, single-file constraint)

The entire model, training loop, data loading, evaluation, and serialization live in one file. The challenge rules require all code in `train_gpt.py` (hard limit: 1500 lines).

**Model (GPT class):** Transformer with RMSNorm, RoPE, Grouped Query Attention (GQA), ReLU²/LeakyReLU(0.5)² MLP (`LEAKY_SLOPE`), tied embeddings, logit softcapping, and skip connections between layers.

**Optimizer:** Muon (Newton-Schulz orthogonalization) for 2D matrix parameters; Adam for embeddings and scalar/control parameters. Separate learning rate groups for embeddings, matrices, scalars, and optional untied head.

**Data pipeline:** Binary shards (256-int header + uint16 tokens) → `TokenStream` → `DistributedTokenLoader` → sequential streaming batches. No random sampling.

**Evaluation:** Tokenizer-agnostic BPB metric computed via SentencePiece byte-accounting lookup tables, handling token boundaries and leading spaces correctly.

**Serialization:** Mixed int5 (MLP) / int6 (attention) quantization with GPTQ-lite per-row clip search, FP16 passthrough for embeddings + control tensors, zstd-22 compression. 3% magnitude pruning before quantization. Final artifact must be ≤16,000,000 bytes.

### train_gpt_mlx.py

MLX port for Apple Silicon development. Same architecture, different backend.

## Challenge Rules (key constraints)

- Artifact = `len(open("train_gpt.py").read().encode()) + len(compressed_model_bytes)` ≤ 16MB
- **Two separate 10-minute limits:**
  - Training: ≤10 min wallclock on 8×H100s (`MAX_WALLCLOCK_SECONDS=600`)
  - Evaluation (TTT + sliding window): ≤10 min ADDITIONAL (NOT included in training time)
  - Total allowed: up to 20 min (10 train + 10 eval)
- Cannot access validation data during training (test-time training on already-evaluated tokens is allowed)
- TTT must be "score-first": evaluate tokens before training on them
- New SOTA requires ≥0.005 nats BPB improvement with p < 0.01 statistical significance
- Default config: 1024 vocab (SentencePiece BPE), 10 layers, 512 dim, 8 heads, 4 KV heads
- Current best: 1.1492 BPB (10L, VR+GA+XSA4+SWA+LateQAT, 15.3MB artifact)
- SOTA on GitHub (verified, rule-compliant): ~1.067 BPB (PR #462: SwiGLU + AdamW TTT 10ep)
- SOTA on GitHub (unverified/borderline): ~0.978 BPB (PR #517: 100ep Cosine TTT, violates eval time limit)

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
./run_on_runpod.sh --save-log <tag>  # Save full log
./run_on_runpod.sh --upload     # Upload train_gpt.py to pod
./run_on_runpod.sh --rerun      # Re-launch training (upload code + restart)
./run_on_runpod.sh --prep-data [N]   # Download N shards locally (once)
./run_on_runpod.sh --upload-data     # Upload local data to pod
./run_on_runpod.sh --stop       # Stop pod
./run_on_runpod.sh --delete     # Delete pod
```

### Training env vars (inline)
Pass `KEY=VALUE` args directly — forwarded to training process:
```bash
./run_on_runpod.sh EMA_ENABLED=1 SWA_ENABLED=0
./run_on_runpod.sh --rerun TTT_ENABLED=1 TTT_OPTIMIZER=adamw TTT_EPOCHS=10
./run_on_runpod.sh --rerun NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=10240
```

### GPU config
```bash
GPU_COUNT=8 BID_PRICE=1.75 ./run_on_runpod.sh           # 8xH100 spot ($14/hr)
GPU_COUNT=1 BID_PRICE=1.75 ./run_on_runpod.sh           # 1xH100 spot ($1.75/hr)
GPU_ID="NVIDIA RTX PRO 4500 Blackwell" BID_PRICE=0.27 ./run_on_runpod.sh  # cheap size test
```

### Local data (separate from repo)
Data lives at `$LOCAL_DATA_ROOT` (default: `~/dev/personal/parameter-golf-data/`).
```bash
./run_on_runpod.sh --prep-data 1    # Download 1 shard locally (quick iteration)
./run_on_runpod.sh --prep-data 80   # Download all 80 shards (full training)
```
When local data exists, `./run_on_runpod.sh` auto-detects and rsync's it to the pod instead of downloading from HuggingFace. Override path: `LOCAL_DATA_ROOT=/path/to/data ./run_on_runpod.sh`

### Fast experiment workflow (~30s between runs)
```bash
./run_on_runpod.sh --prep-data 1          # Once: download data locally
GPU_COUNT=1 ./run_on_runpod.sh            # Create pod (auto-uploads local data)
./run_on_runpod.sh --save-log "baseline"  # Save results
./run_on_runpod.sh --rerun EMA_ENABLED=1  # New experiment (uploads code, restarts)
./run_on_runpod.sh --save-log "ema"       # Save results
./run_on_runpod.sh --delete               # Clean up
```

### Logging
Save every training run's log after completion:
```bash
./run_on_runpod.sh --save-log "11L_VR1_GA1_prune3pct"
```
This saves to `logs/<timestamp>_<tag>.log` and `logs/<timestamp>_<tag>.summary` with key metrics extracted.

### Cost-saving tips
- **Always delete pods after saving logs/results** — `--save-log <tag>` then `--delete`
- **Use `--rerun` to iterate** — skips pod creation + data download, ~30s turnaround
- **Pre-download data locally** — `--prep-data 1` once, auto-uploaded to every pod
- **Test artifact size on cheap GPUs** — RTX PRO 4500 spot ($0.27/hr) before H100. Needs smaller batch:
  `GPU_ID="NVIDIA RTX PRO 4500 Blackwell" BID_PRICE=0.27 ./run_on_runpod.sh TRAIN_BATCH_TOKENS=131072 TRAIN_SEQ_LEN=1024 EVAL_STRIDE=0 EMA_ENABLED=0`
- **Use `EVAL_STRIDE=0`** to skip sliding window eval on single GPU
- **Use `EMA_ENABLED=0`** on single GPU — EMA kills throughput (~32% slower)
- **Always `--stop` or `--delete` pods when done** — spot 8xH100 is $14/hr
- **Spot instances get preempted** — always use `nohup` and check pod status
- **TTT needs H100** — OOMs on 32GB GPUs. Only enable on H100+
- **TTT on single GPU is very slow** — use 8xH100 for TTT experiments
- **TTT has separate 10-min eval budget** — not counted in training time. ~20 epochs safe (~380s TTT + ~200s eval)
- **TTT adapts all params by default** — Muon for 2D + AdamW for 1D (when `TTT_OPTIMIZER=muon`)
- **TTT cosine LR enabled by default** (`TTT_COSINE=1`) — prevents overfitting at high epoch counts
- **Check pod status every 60s during experiments** — spot pods get preempted, don't waste money on dead pods
- **Save logs after EVERY experiment** before starting the next one — logs are lost when pod dies
