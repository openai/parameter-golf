# Hardik Top5 Run

Draft submission package for the OpenAI Parameter Golf `track_10min_16mb` track.

This folder contains a self-contained copy of the current best local training script, prepared in the expected `records/...` format for pull request submission. The copied `train_gpt.py` has been patched so its default dataset and tokenizer paths resolve from the repository root even when executed from inside this records folder.

## Status

- Submission structure: ready
- Script packaging: ready
- Relative-path cleanup: ready
- Reproducibility notes: ready
- Final leaderboard claim: pending a fresh logged run for this exact script

## Architecture Summary

The current model is a Parameter Golf "podium build" based on the LeakyReLU^2 + TTT + Parallel Muon family, with the following default stack:

- Vocabulary: SentencePiece 8192-token model
- Backbone: 11 transformer layers, 512 hidden dim, 8 attention heads, 4 KV heads
- MLP: 3.0x expansion with LeakyReLU(0.5)^2 activation
- Residual layout: parallel residual attention + MLP path
- Recurrence: block recurrence enabled by default on layers `4,5` with `RECURRENCE_LOOPS=3`
- Attention extras: QK gain, partial RoPE (`ROPE_DIMS=16`), XSA on the last 4 layers
- Token enrichments: Bigram hash embedding and shared value embeddings
- Optimizers: AdamW for token/scalar groups + custom Parallel Muon for matrix banks
- Averaging: EMA by default, optional SWA/LAWA
- Compression path: mixed int6 / int8 quantization with lzma export
- Eval extras: sliding-window validation and optional legal score-first TTT

## Default Model Size

- Parameter count (default config): `31,581,276`
- Code bytes for this packaged `train_gpt.py`: `97,310`

Note: the contest artifact limit is code bytes plus compressed model bytes. This folder does not yet include a verified compressed artifact size for the exact packaged script because a fresh training/eval run has not been logged for this copy yet.

## Innovations Used

1. SP8192 tokenizer defaults
2. Depth recurrence through repeated middle blocks
3. Parallel residual transformer blocks
4. Learned QK gain scaling
5. Parallel Muon / MuonEq-R optimizer path
6. Hessian SDClip for GPTQ-style clipping
7. GPTQ-style embedding quantization
8. Optional legal score-first TTT with SGD or Adam

## Hardware Used

- Packaging/validation of this submission folder: local Windows machine
- Target contest hardware: 8x H100 80GB SXM
- Final authoritative leaderboard run hardware for this exact script: `TBD`

## Training Time

- Default script wallclock cap: `600` seconds (`MAX_WALLCLOCK_SECONDS=600`)
- Fresh measured 8xH100 runtime for this packaged copy: `TBD`

## Achieved Score

- Fresh logged `val_bpb` for this packaged copy: `TBD`
- Fresh logged `val_loss` for this packaged copy: `TBD`
- Verified total submission bytes for this packaged copy: `TBD`

Do not claim a leaderboard score from this folder until `train.log` and `submission.json` are updated from a real run of the included script.

## Reproducibility

The script is designed to be configurable through environment variables and avoids absolute machine-specific paths.

- Seed default: `1337`
- Dataset default: resolved from repo root as `data/datasets/fineweb10B_sp8192`
- Tokenizer default: resolved from repo root as `data/tokenizers/fineweb_8192_bpe.model`
- Optional acceleration imports (`flash_attn_interface`, `zstandard`) have safe fallbacks
- No network calls are made during training or evaluation

## Run From This Folder

From repository root:

```bash
cd records/track_10min_16mb/hardik_top5_run
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Example with explicit paths:

```bash
cd records/track_10min_16mb/hardik_top5_run
DATA_PATH=../../../data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_8192_bpe.model \
RUN_ID=hardik_top5_run_seed1337 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## What To Update Before PR

1. Run the packaged script on the intended hardware.
2. Replace the placeholder `train.log` with the real training log.
3. Update `submission.json` with real `val_loss`, `val_bpb`, and `bytes_total`.
4. If submitting as a new record, include enough independent seeds to satisfy the repo significance rule.

## Notes

- This folder intentionally mirrors the structure of existing successful records in `records/track_10min_16mb/`.
- The root-level `train_gpt.py` remains your active development file; this folder is the frozen submission copy for PR review.
