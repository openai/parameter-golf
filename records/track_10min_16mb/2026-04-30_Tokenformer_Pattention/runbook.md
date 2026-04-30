# Tokenformer Pattention — Remote H100 Runbook

This runbook drives the remote validation passes referenced in `README.md`. The 2026-04-30
submission was executed against an NVIDIA Brev 8×H100 SXM pod. The dataset paths and command
shapes below are exactly what produced the `logs_8xh100/` files in this folder.

## What this submission's runs were (recorded for reproducibility)

The submission's headline comparison is a two-run head-to-head: dense baseline (Run 1) vs
matched-params Pattention (Run 2). Both runs come from a single 8×H100 SXM pod
(`uneven-bronze-albatross`, NVIDIA H100 80GB HBM3, torch 2.11.0+cu128, FA3, Ubuntu 22.04,
python 3.10), driven by `runs/run_3.sh` whose two relevant invocations are:

```bash
base_env="PYTHONUNBUFFERED=1 VOCAB_SIZE=1024 SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
          ITERATIONS=20000 VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=200 \
          MASTER_ADDR=127.0.0.1 MASTER_PORT=29500"

# Run 1: dense baseline control
env $base_env PATTENTION=0 MLP_MULT=2 RUN_ID=run1_baseline_pat0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py > runs/run1_baseline_pat0.log 2>&1

# Run 2: matched-params Pattention (the submission entry)
env $base_env PATTENTION=1 MLP_MULT=2 PATTENTION_P_RATIO=1.0 RUN_ID=run2_pat_matched \
  torchrun --standalone --nproc_per_node=8 train_gpt.py > runs/run2_pat_matched.log 2>&1
```

A third supplementary pilot was also captured in the same session (`PATTENTION=1 MLP_MULT=4`,
26.5M params, 11.36 MiB compressed, val_bpb 1.3746). It is preserved at
`logs_8xh100/run3_pat_mlp4.log` for reference but is not part of the headline comparison.

Total wallclock for runs 1+2: 23:10 (11:30 + 11:40). Submission entry is Run 2.

The remainder of this file is the original prep/sanity/3-seed runbook used during development.
Both assume you have an H100 pod up with the repo cloned and dataset already downloaded.

## 0. Pod prep (one-time, on the remote machine)

```bash
cd /workspace
git clone https://github.com/<your fork>/parameter-golf.git || true
cd parameter-golf
# If the data isn't already present on the pod template:
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This populates `./data/datasets/fineweb10B_sp1024/` (full 80-shard train + full val) and
`./data/tokenizers/fineweb_1024_bpe.model`. **Do not run with fewer than the full 80 shards** —
the depth-recurrence non-record write-up showed a ~0.1 bpb regression from training on 1 shard.

Replace `train_gpt.py` at the repo root with the one in this folder (or just check out the branch
that contains the Pattention diff).

## 1. 1xH100 sanity check (~3 min)

Confirms the run starts cleanly, that gradients flow through Pattention, and that the int8+zlib
roundtrip pipeline produces a sane `final_int8_zlib_roundtrip val_bpb`. Use a short wallclock so
this is cheap on a 1xH100 pod.

```bash
RUN_ID=tokenformer_pat_sanity_1xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
PATTENTION=1 PATTENTION_P_RATIO=1.0 \
MAX_WALLCLOCK_SECONDS=180 \
WARMUP_STEPS=10 \
VAL_LOSS_EVERY=200 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 train_gpt.py \
  2>&1 | tee logs/tokenformer_pat_sanity_1xh100.console.log
```

Acceptance:
- `model_params:` line reports ~17,050,696.
- `pattention_enabled:True pattention_p_ratio:1.0` line is present.
- Train loss strictly decreases (allow a small step-2 transient on the first ~5 steps from the
  zero-init residual stream).
- `final_int8_zlib_roundtrip val_bpb:` is finite and within +0.05 of the pre-quant `val_bpb`.

## 2. 8xH100 3-seed submission run (~10 min × 3)

These are the runs whose logs go in the submission folder. Use the canonical seeds.

```bash
for SEED in 1337 7 42; do
  RUN_ID=tokenformer_pattention_seed${SEED} \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  PATTENTION=1 PATTENTION_P_RATIO=1.0 \
  SEED=${SEED} \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
    2>&1 | tee logs/tokenformer_pattention_seed${SEED}.console.log
done
```

The script writes its own log to `logs/tokenformer_pattention_seed${SEED}.txt` (it always emits
that path on the first stdout line of the run). The full code text is prepended at log start so
each `*.txt` is a self-contained submission artifact.

## 3. Optional baseline ablation (3-seed dense MLP for the README table)

```bash
for SEED in 1337 7 42; do
  RUN_ID=baseline_dense_mlp_seed${SEED} \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  PATTENTION=0 \
  SEED=${SEED} \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
    2>&1 | tee logs/baseline_dense_mlp_seed${SEED}.console.log
done
```

## 4. Collect the artifacts back into the records folder

After the 8xH100 runs land:

```bash
mkdir -p records/track_10min_16mb/2026-04-30_Tokenformer_Pattention
for SEED in 1337 7 42; do
  cp logs/tokenformer_pattention_seed${SEED}.txt \
     records/track_10min_16mb/2026-04-30_Tokenformer_Pattention/train_seed${SEED}.log
done
```

Then update `submission.json` with the chosen reporting seed's `val_bpb`, `val_loss`, and the
final int8+zlib bytes (printed on the `serialized_model_int8_zlib:` log line + this script's own
size, summed below the `bytes_total` field — the README in the repo root explains the accounting).

The `bytes_total` field should be:
- code bytes = `len(open(records/.../train_gpt.py, 'rb').read())`
- compressed model bytes = the `serialized_model_int8_zlib:N bytes` line in the train log
- `bytes_total = code + compressed_model`. Must be ≤ 16,000,000 bytes (decimal).

## Tip: one-shot driver from a workstation

If you'd rather drive everything from your laptop, an `ssh user@pod` wrapper around section 2
that streams the logs back is the cleanest path. Each seed produces `logs/<run_id>.txt` plus
`logs/<run_id>_model.npz` and `logs/<run_id>_model.int8.ptz` (the latter is the compressed
artifact tracked against the 16 MB cap).
