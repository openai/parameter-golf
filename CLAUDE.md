# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

OpenAI's "Parameter Golf" challenge: train the best language model that fits in a **16MB artifact** (code + compressed weights) and trains in **under 10 minutes on 8×H100s**. Scored by bits-per-byte (val_bpb) on a fixed FineWeb validation set. Inspired by NanoGPT Speedrunning but optimizing L(N) — lowest loss for fixed parameter count.

- **Baseline**: 9 layers, dim 512, vocab 1024, ~17M params → val_bpb 1.2244
- **SOTA merged**: 10 layers, FP16 embed, Muon WD, OvertoneInit → val_bpb 1.1748
- **Frontier (open PRs)**: 11L, int6 QAT, SWA, SmearGate, MLP 3× → val_bpb ~1.13
- **Our consensus stack**: 11L, dim 512, MLP 3×, ~26.5M params, int6 QAT+SWA+SmearGate → artifact 4.3MB (int6+zlib)

## Key Commands

### Data Download
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
# Smaller subset for local iteration
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

### Training (CUDA, RunPod / H100)
```bash
RUN_ID=consensus_v1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
Use `--nproc_per_node=8` for 8×H100. Override `MAX_WALLCLOCK_SECONDS=0` to remove the 10-minute cap.

### Training (MLX, Apple Silicon)
```bash
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=16384 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=16384 GRAD_ACCUM_STEPS=4 python3 train_gpt_mlx.py
```
Note: with seq_len=2048, minimum TRAIN_BATCH_TOKENS must be ≥2048. Use GRAD_ACCUM_STEPS=4 to fit in 16GB.

### RunPod Automation
```bash
source .env  # loads RUNPOD_API_KEY
./scripts/runpod.sh create 1       # create 1×H100 pod (~$3/hr)
./scripts/runpod.sh setup           # clone repo + download data
./scripts/runpod.sh run consensus_v1  # run training
./scripts/runpod.sh fetch consensus_v1  # get results locally
./scripts/runpod.sh terminate       # delete pod
```

## Architecture

Both training scripts implement the **consensus stack** (hard cap: 1500 lines each):

- **`train_gpt.py`** (~1460 lines) — PyTorch/CUDA for RunPod/H100. GPT model with: 11 layers, MLP 3×, GQA, RoPE, SmearGate, logit softcap (20), tied embeddings. Muon optimizer with WD=0.038. Int6 QAT (STE), SWA during warmdown, FP16 embedding passthrough. Int6 step=4 quantization + zlib compression. LoRA TTT at eval. DDP multi-GPU.

- **`train_gpt_mlx.py`** (~1214 lines) — MLX port for local Apple Silicon iteration. Same consensus stack, adapted optimizers, eager eval mode for 16GB machines.

- **`data/`** — Dataset download and retokenization. Shards in `data/datasets/`, tokenizers in `data/tokenizers/`.

- **`records/`** — Submission history. Two tracks: `track_10min_16mb/` (leaderboard) and `track_non_record_16mb/` (unlimited compute).

- **`scripts/`** — Automation tools. `runpod.sh` for RunPod pod lifecycle (create/setup/run/fetch/terminate) using REST API with curl.

## Consensus Stack (implemented)

| Technique | Env Var | Default | Status |
|-----------|---------|---------|--------|
| 11 layers | `NUM_LAYERS` | 11 | Implemented |
| MLP 3× | `MLP_MULT` | 3 | Implemented |
| Seq 2048 | `TRAIN_SEQ_LEN` | 2048 | Implemented |
| SmearGate | — | always on | Implemented |
| Logit softcap 20 | `LOGIT_SOFTCAP` | 20 | Implemented |
| Muon WD 0.038 | `MUON_WEIGHT_DECAY` | 0.038 | Implemented |
| Int6 QAT (STE) | `QAT_ENABLED`, `QAT_BITS` | 1, 6 | Implemented |
| SWA | `SWA_ENABLED`, `SWA_EVERY` | 1, 50 | Implemented |
| FP16 embed passthrough | `FP16_EMBED_PASSTHROUGH` | 1 | Implemented |
| Int6 quantization | `QUANT_BITS` | 6 | Implemented |
| OrthoInit + muP | — | — | Not yet |
| BigramHash | — | — | Not yet |
| Sliding window eval | — | — | Not yet |

## Knowledge Base

Research and analysis in `docs/`:

- **`docs/README.md`** — Problem definition, leaderboard, submission analysis (merged vs PR frontier), R&D directions across 3 tiers
- **`docs/nanogpt-speedrun.md`** — 77 records from modded-nanogpt. Transferable: SmearGate, BigramHash, NorMuon, value embeddings, sliding window, U-net skips
- **`docs/nanogpt-slowrun.md`** — 27 records across 3 tracks. Key: heavy regularization, value projections from x0, per-head gating, layer looping, EMA/SWA
- **`docs/small-model-research.md`** — Sub-100M model techniques: MobileLLM, Depth Delusion, RingFormer, QAT, BitNet, optimizer advances

Reference subtrees: `modded-nanogpt/`, `slowrun/`

## Experiment Logs

Local runs are stored in `logs/` (gitignored). Each run has a subdirectory with README.md, training log, and model artifacts.

| Run | Params | Iters | val_bpb | Artifact | Notes |
|-----|--------|-------|---------|----------|-------|
| mlx_smoke_baseline | 17M | 200 | 2.3244 | 10.1MB int8 | Baseline, Apple Silicon |
| consensus_smoke_int8 | 26.5M | 10 | 3.6078 | 8.6MB int8 | Consensus stack, int8 quant |
| consensus_smoke_int6 | 26.5M | 10 | 3.6285 | **4.3MB int6** | Consensus stack, int6 quant |

## Submission Rules

- New SOTA must beat existing by ≥0.005 nats at p < 0.01 (typically 3 run logs)
- Artifact size = code bytes + compressed model bytes ≤ 16,000,000 (decimal, not MiB)
- Submissions are PRs that add a folder under `records/` with: `README.md`, `submission.json`, `train_gpt.py`, `train.log`
- No external downloads or network calls allowed during evaluation
- Eval time limit: 10 minutes on 8×H100 (separate from training time)

## Key Hyperparameters (env vars)

All configured via environment variables. See the `Hyperparameters` class in each script for the full list. Key new ones: `QAT_ENABLED`, `QAT_BITS`, `SWA_ENABLED`, `SWA_EVERY`, `SWA_BLEND_FINAL`, `MUON_WEIGHT_DECAY`, `QUANT_BITS`, `FP16_EMBED_PASSTHROUGH`.

## Dependencies

Core: `torch`, `numpy`, `sentencepiece`, `tqdm`. MLX path adds `mlx`. Data scripts need `huggingface-hub`, `datasets`. RunPod automation needs `curl`, `jq`. See `requirements.txt`.
