# Parameter Golf, in plain English

This repo is for a challenge: build the best language model that fits inside a `16 MB` artifact and trains in under `10 minutes` on `8xH100`.

The score is not just the raw validation loss. The important number is the model’s `final_int8_zlib_roundtrip_exact val_bpb`, which means:

1. train the model
2. compress the weights to int8
3. zip them
4. decompress and load them back
5. score the round-tripped model on FineWeb

Lower `BPB` is better.

## The main files

- `[train_gpt.py](/Users/sanketdongre/Documents/Projects/parameter-golf/train_gpt.py)` is the PyTorch training script for CUDA GPUs.
- `[train_gpt_mlx.py](/Users/sanketdongre/Documents/Projects/parameter-golf/train_gpt_mlx.py)` is the Mac / Apple Silicon version for local smoke testing.
- `[data/cached_challenge_fineweb.py](/Users/sanketdongre/Documents/Projects/parameter-golf/data/cached_challenge_fineweb.py)` downloads the challenge data.
- `[scripts/pg_lab.py](/Users/sanketdongre/Documents/Projects/parameter-golf/scripts/pg_lab.py)` prints run commands, compares logs, and packages a finished run into a submission folder.
- `[docs/submission-playbook.md](/Users/sanketdongre/Documents/Projects/parameter-golf/docs/submission-playbook.md)` explains the submission workflow.

## What the model is doing

The model is a small Transformer. In simple terms:

- it turns text into tokens
- tokens go through a stack of transformer blocks
- the model predicts the next token
- training adjusts the weights so the predictions get better

The default baseline in this repo is intentionally small:

- `9` layers
- `512` hidden dimension
- `8` attention heads
- `4` KV heads
- tied input/output embeddings
- `1024` token vocabulary

## Why compression matters

This challenge is really about squeezing quality into a tiny artifact.

That means two things matter:

- the model must learn well during training
- the saved weights must survive quantization and compression with as little damage as possible

That is why this PR added:

- compression-focused logging in the trainer
- a helper to compare pre-quant and post-quant runs
- a packaging command that refuses smoke logs and incomplete logs

## How to run things

For a quick local smoke test on a Mac, use the MLX script.
For a real scoring run, use `train_gpt.py` on CUDA hardware.

The helper command:

```bash
python3 scripts/pg_lab.py command --profile cuda-baseline-1x --stage base --focus ref
```

prints the baseline command with the right environment variables.

## What changed in this branch

This branch is not the scored submission itself. It prepares the repo for one.

It adds:

- stricter record packaging
- exact-delta log parsing
- safer smoke-run handling
- clearer docs for experiments and grants

## The short version

If you only remember three things:

1. the challenge is about `score × size × time`
2. the score used for submission is the post-compression roundtrip score
3. this branch prepares the workflow, but the actual winner still needs a real GPU run
