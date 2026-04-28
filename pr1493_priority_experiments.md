# PR1493 Priority Experiments Runbook

This is an experiment harness for finding the next major win on top of PR #1493 plus the current best TTT sweep. The goal is attribution first: run one change at a time, compare against the same seed/settings, then stack only confirmed wins.

## Current Comparator

Use the tuned TTT baseline as the comparator, not raw PR #1493:

```text
PR1493 reproduced, seed 42, QK_GAIN_INIT=5.25:
  quantized_ttt: 1.08103358

Best local TTT sweep so far:
  TTT_LR=0.007, TTT_EPOCHS=5
  quantized_ttt: 1.08079274
```

Anything that does not beat roughly `1.08079` on the same seed is not a real win. To matter for the leaderboard acceptance margin, we likely need another `~0.0017-0.0020 BPB`, not just noise-level movement.

## Implemented Switches

All changes are off by default in `train_pr1493.py`.

| Experiment | Env |
| --- | --- |
| Document/BOS-aware training loader | `DOC_SHUFFLE_ENABLED=1` |
| Weight decay schedule | `WD_SCHEDULE_ENABLED=1` |
| IHA q/k head mixing | `IHA_ENABLED=1` |
| IHA q/k/v head mixing | `IHA_ENABLED=1 IHA_MIX_V=1` |
| Train-only MTP auxiliary loss | `MTP_WEIGHT=0.10 MTP_STEPS=1` |
| Eval-only extra recurrence | `EVAL_NUM_LOOPS=3` |

MTP is intentionally disabled during TTT adaptation inside `eval_val_ttt`. We want to test whether MTP improves the trained model, not whether TTT can optimize a mixed next-token plus t+2 objective.

## SSH Setup

Use the same environment family as the challenge and prior Modal scripts:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.9.1
pip install numpy sentencepiece huggingface-hub datasets tqdm brotli psutil packaging ninja wheel setuptools
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
```

Download the SP8192 matched dataset:

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

Sanity check:

```bash
ls data/datasets/fineweb10B_sp8192/fineweb_train_*.bin | wc -l
ls data/datasets/fineweb10B_sp8192/fineweb_val_*.bin | wc -l
ls -lh data/tokenizers/fineweb_8192_bpe.model
```

Expected: 128 train shards, validation shards present, and `fineweb_8192_bpe.model` present.

## Baseline Command

Run this once on the SSH machine if you need to confirm machine parity:

```bash
RUN_ID=pr1493_baseline_ttt_s42 \
SEED=42 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 \
TTT_LR=0.007 \
TTT_EPOCHS=5 \
torchrun --standalone --nproc_per_node=8 train_pr1493.py
```

Expected shape:

```text
pre-quantization post-ema val_bpb: around 1.0875-1.0880
quantized_sliding_window val_bpb: around 1.083
quantized_ttt val_bpb: around 1.08079-1.08103
```

If this baseline is materially worse, stop and debug environment drift before testing new ideas.

## Run Order

Run one experiment at a time:

```bash
# 1. Document-aware loader
RUN_ID=pr1493_docshuffle_s42 SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
DOC_SHUFFLE_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_pr1493.py

# 2. Weight decay schedule
RUN_ID=pr1493_wd_s42 SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
WD_SCHEDULE_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_pr1493.py

# 3. IHA q/k only
RUN_ID=pr1493_iha_s42 SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
IHA_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_pr1493.py

# 4. MTP, conservative first pass
RUN_ID=pr1493_mtp_s42 SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
MTP_WEIGHT=0.10 MTP_STEPS=1 \
torchrun --standalone --nproc_per_node=8 train_pr1493.py

# 5. Eval-only extra recurrence
RUN_ID=pr1493_evalloop3_s42 SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.007 TTT_EPOCHS=5 \
EVAL_NUM_LOOPS=3 \
torchrun --standalone --nproc_per_node=8 train_pr1493.py
```

Parse results:

```bash
rg "pre-quantization|quantized_sliding_window|quantized_ttt|Total submission size|stopping_early" logs/pr1493_*.txt
```

## How To Judge

Primary metric:

```text
quantized_ttt val_bpb
```

Secondary diagnostics:

```text
pre-quantization post-ema val_bpb
quantized_sliding_window val_bpb
quant gap = quantized_sliding_window - pre-quantization post-ema
train steps reached before wallclock cap
Total submission size
```

Interpretation:

- If pre-quant improves but post-quant worsens, the idea may be making weights less quantizable.
- If sliding improves but TTT does not, the idea may be incompatible with adaptation.
- If TTT improves by less than `0.0002 BPB`, treat it as noise until repeated across seeds.
- If doc shuffle slows training enough to reduce steps, compare against the actual stopped step count before calling the idea bad.
- If eval-only recurrence helps non-TTT but hurts TTT, do not stack it with TTT.

## Critical Caveats

The current `train_pr1493.py` is an experiment script, not a final legal submission. It is larger than the original script, so final submission must be minified/packed and size-checked again.

Known old baseline logs had:

```text
Code size: 48583 bytes
Serialized model quantized+brotli: about 15.97 MB
Total submission size: slightly over 16 MB before packing
```

So do not assume a good BPB run is submit-ready until:

```bash
rg "Total submission size" logs/<run_id>.txt
```

and the final packed submission is checked under `16,000,000` bytes.

## Modal Fallback

`run_pr1493_modal.py` exposes the same experiments:

```bash
modal run run_pr1493_modal.py --experiment docshuffle --seed 42
modal run run_pr1493_modal.py --experiment wd --seed 42
modal run run_pr1493_modal.py --experiment iha --seed 42
modal run run_pr1493_modal.py --experiment mtp --seed 42
modal run run_pr1493_modal.py --experiment evalloop3 --seed 42
```

The attempted Modal launch built the image but was blocked by workspace billing limits on both tested profiles, so SSH is currently the practical path.
