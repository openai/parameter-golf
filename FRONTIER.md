# Frontier Recipe Workspace

This repo now has a dedicated frontier experimentation path built around [train_gpt_frontier.py](train_gpt_frontier.py), while leaving the beginner baselines and archived `records/` snapshots untouched.

## What This Path Covers

The frontier trainer exposes the public-best recipe family as explicit environment-variable toggles:

- mixed quantization by tensor group: `QUANT_MLP_BITS`, `QUANT_ATTN_BITS`, `QUANT_BIGRAM_BITS`
- SWA controls: `SWA_ENABLED`, `SWA_START_FRAC`, `SWA_EVERY`
- optimizer regularization: `ADAMW_WEIGHT_DECAY`, `MUON_WEIGHT_DECAY`
- warmdown and training length: `WARMDOWN_ITERS`, `ITERATIONS`, `MAX_WALLCLOCK_SECONDS`
- BigramHash size: `BIGRAM_VOCAB_SIZE`, `BIGRAM_DIM`
- architecture shape: `NUM_LAYERS`, `MODEL_DIM`, `MLP_MULT`, `NUM_HEADS`, `NUM_KV_HEADS`
- tied embeddings and architectural extras: `TIE_EMBEDDINGS`, `SKIP_CONNECTIONS_ENABLED`, `SMEAR_ENABLED`, `ORTHO_INIT_ENABLED`
- evaluation stride: `EVAL_STRIDE`
- artifact compression: `COMPRESSOR`

The defaults are rules-safe and self-contained:

- no seed-search workflow
- no external checkpoints
- no hidden offline optimization machinery
- no evaluation-time downloads

## Preset Ladder

These presets are defined in [research/presets.py](research/presets.py):

- `baseline`: 9L public-frontier baseline rung
- `frontier_partial_quant`: isolate int5 MLP with int6 attention/bigram
- `frontier_partial_swa_wd`: isolate WD=0.04 + warmdown=3000 + SWA
- `frontier_partial_bigramhash`: isolate the first hash-table expansion
- `frontier_arch_10l_variant`: isolate the 10th layer
- `frontier_combined_public_like`: public-best-inspired combined recipe
- `nearby_variant_1`: public-like recipe with smaller BigramHash for byte headroom
- `nearby_variant_2`: public-like recipe with later SWA entry
- `local_frontier_proxy_mlx`: Apple Silicon proxy for local-first screening of frontier-adjacent ideas

Each run records the preset name, git state, resolved env, and config diff from its baseline when applicable.

## Local-First Run Scales

Operational run scales are also defined in [research/presets.py](research/presets.py):

- `smoke`
- `probe_short`
- `probe_medium`
- `long_local_overnight`
- `long_local_24h`

These scales overlay iteration or wallclock budget, approximate validation limits, checkpoint cadence, summary cadence, and whether to pay the final quantized roundtrip cost.

## Commands

List presets:

```bash
python3 research/run.py --list
```

List run scales:

```bash
python3 research/run.py --list-scales
```

Frontier CUDA preflight:

```bash
python3 scripts/check_env.py --target cuda --require zstandard
python3 scripts/check_frontier_env.py
python3 scripts/check_data.py --data-path ./data/datasets/fineweb10B_sp1024 --tokenizer-path ./data/tokenizers/fineweb_1024_bpe.model --min-train-shards 1 --seq-len 2048
```

On RunPod / shared PyTorch images, do not reinstall torch on top of the image. Use [CLOUD_SETUP.md](CLOUD_SETUP.md) before launching frontier presets.

Run the frontier baseline on CUDA:

```bash
python3 research/run.py --preset baseline --run-name baseline_9l --nproc-per-node 1
```

Run the public-like combined recipe:

```bash
python3 research/run.py --preset frontier_combined_public_like --run-name public_like --nproc-per-node 8
```

Compare frontier runs only:

```bash
python3 research/compare_runs.py --family frontier
```

Run a local frontier proxy probe on Apple Silicon:

```bash
python3 research/run.py --preset local_frontier_proxy_mlx --scale probe_short --run-name local_probe
```

Resume an interrupted local run:

```bash
python3 research/run.py --resume-run-dir research/results/runs/<timestamp>_<run_name>
```

## Practical Use

Recommended workflow:

1. Use the existing MLX path from [WORKSPACE.md](WORKSPACE.md) for local smoke checks.
2. Use the frontier presets on CUDA for leaderboard-relevant comparisons.
3. Promote only the best ideas into 8xH100-style runs.
