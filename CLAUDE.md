# Parameter Golf — Project Context

## What This Is

OpenAI's Parameter Golf competition (March 18 - April 30, 2026): train the best language model that fits in a 16MB artifact, training under 10 minutes on 8xH100 GPUs. Metric is val_bpb (bits per byte) on FineWeb — lower is better.

## Local Setup

- **Hardware**: Single RTX 3070 (8GB VRAM) on WSL2
- **Python**: 3.12, venv at `.venv/`
- **Adapted script**: `train_gpt_3070.py` — modified baseline for single GPU with 500 iterations
- **Original baseline**: `train_gpt.py` — do not modify this file

## Key Constraints

- 16MB total artifact (code + compressed model)
- val_bpb metric (tokenizer-agnostic, lower is better)
- Baseline: 9 layers, 512d, 8 heads, 4 KV heads, 2x MLP, vocab 1024, tied embeddings
- Current SOTA: 1.1194 bpb (baseline: 1.2244)

## Development Workflow

1. All local experiments use `train_gpt_3070.py` (not the original)
2. Run with `python3 train_gpt_3070.py` (no torchrun needed)
3. Use env vars to override: `ITERATIONS=200 python3 train_gpt_3070.py`
4. If OOM, increase `GRAD_ACCUM_STEPS` (default 64, try 128)
5. See `SETUP_PLAN.md` for the full incremental replication plan

## Data

Download with: `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10`
Data goes to `./data/datasets/fineweb10B_sp1024/`

## Autoresearch

Karpathy's autonomous ML experiment loop adapted for this competition.
See `autoresearch/program.md` for the adapted research agenda.
Reference repo cloned at `/mnt/c/dev/autoresearch/`.

## Known Winning Techniques (in order of impact)

1. Sliding window eval (stride=64) — free -0.032 bpb
2. FP16 embedding passthrough — free -0.005 bpb
3. 3x MLP + more layers (11L) — large gain
4. Int6 QAT with STE — eliminates quantization gap
5. SmearGate + BigramHash — bigram context
6. EMA weight averaging (decay=0.997)
7. LeakyReLU(0.5)^2 — one-line activation swap
8. XSA on last 3-4 layers
9. Test-time training (LoRA TTT)

## Known Failures (do not attempt)

- MoE (0.06-0.08 bpb worse at this scale)
- SwiGLU (45% slower, net negative)
- Depth recurrence (+0.025 bpb worse)
- Factored embeddings with vocab 1024
