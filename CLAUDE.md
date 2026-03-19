# Parameter Golf

## Project

OpenAI's Parameter Golf challenge: train the best language model that fits in a 16MB artifact (code + int8+zlib compressed weights), under 10 minutes on 8xH100 SXM. Scored by `val_bpb` (bits per byte) on FineWeb validation set.

## Baseline Features (upstream train_gpt.py)

The upstream script already includes:
- **MTP** (multi-token prediction) — 1 auxiliary head, weight 0.1, trained via Muon, stripped on export
- **EMA** — exponential moving average of weights (decay 0.999), used for export
- **QAT** — fake-quantize weights during last 15% of training for int8 robustness
- **FP16 embedding export** — keeps tok_emb in fp16 instead of int8
- **Sliding window eval** — optional (`EVAL_STRIDE>0`), scores with near-full context
- **Magnitude pruning** — optional (`PRUNE_FRACTION>0`), zeros small weights for better zlib
- **Int6 quantization** — optional (`INT6_QUANT=1`), narrower range for better compression
- **Configurable MLP hidden dim** — `MLP_HIDDEN` decoupled from `MLP_MULT`

Key env vars: `MTP_NUM_HEADS` (default 1), `MTP_LOSS_WEIGHT` (default 0.1), `EMA_ENABLED` (default 1), `EMA_DECAY` (default 0.999), `QAT_FRACTION` (default 0.15), `EVAL_STRIDE` (default 0), `EVAL_SEQ_LEN`, `PRUNE_FRACTION` (default 0), `INT6_QUANT` (default 0), `FP16_EMBED_EXPORT` (default 1)

## Experiment Results (1xH100 SXM, 10 min cap)

| Run | Steps | val_bpb (post-quant) |
|-----|-------|---------------------|
| Old baseline (no MTP/EMA/QAT) | 1078 | 1.3500 |
| Our MTP K=2 α=0.2 (old code) | 1124 | 1.3430 |
| New baseline (needs testing) | — | — |

## Repo Structure

- `train_gpt.py` — main training script (model, optimizer, eval, serialization — all in one)
- `train_gpt_mlx.py` — Apple Silicon variant
- `data/` — dataset download scripts and tokenizers
- `records/` — leaderboard submissions (README, submission.json, train_gpt.py, train.log each)
- `run_baseline.sh` / `run_mtp.sh` — convenience scripts for RunPod

## Workflow

- Code lives locally and on `sp00mm/parameter-golf` GitHub fork (branch: `mtp-auxiliary-heads`)
- Training runs on RunPod H100 SXM pods — clone from fork, download data, run, grab results, terminate
- Data download: `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10`
- 1xH100 for iteration (~$2.69/hr), 8xH100 SXM for final submission only

## Key Commands (RunPod)

```bash
# Setup
cd /workspace && git clone https://github.com/sp00mm/parameter-golf.git && cd parameter-golf
git checkout mtp-auxiliary-heads
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# New baseline (1xH100) — MTP(1) + EMA + QAT + FP16 embed all on by default
RUN_ID=baseline_v2 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 VAL_LOSS_EVERY=500 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee baseline_v2.log

# Experiment: MTP K=2 + higher weight
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 RUN_ID=mtp2 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 VAL_LOSS_EVERY=500 torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee mtp2.log
```

## Submission Requirements

PR to `openai/parameter-golf` adding a folder under `records/track_10min_16mb/` with:
- `README.md` — explains approach
- `submission.json` — author, github_id (sp00mm), val_bpb, byte sizes
- `train_gpt.py` — standalone script
- `train.log` — from 8xH100 SXM run
- Must beat SOTA by 0.005 nats at p < 0.01 (multiple seed runs needed)

## Leaderboard (as of 2026-03-18)

1. 2048 seq length: val_bpb 1.206
2. fp16 Embed: val_bpb 1.2197
3. Naive Baseline: val_bpb 1.2244
