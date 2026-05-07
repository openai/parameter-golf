# Non-record: 4090 ablations on ValCalib GPTQ + XSA + BigramHash stack

**Author:** Wolfie8935  
**Intent:** Document a **budget hardware** exploration (single **RTX 4090**, 24GB VRAM) on the same code lineage as the 10-minute leaderboard stack ([`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072`](../track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md)), **without** claiming a new 8×H100 record.

This is **not** comparable to the public SOTA (~**1.1147** sliding BPB on 8×H100 SXM): we used **`TRAIN_BATCH_TOKENS=196608`** (OOM-safe on 24GB) instead of the multi-GPU throughput recipe, so step counts and trajectories differ.

## What is in this folder

| File | Purpose |
|------|---------|
| `train_gpt.py` | Unmodified snapshot from the ValCalib / PR #1019 lineage (same as reference record). |
| `requirements.txt` | Same as reference record (FlashAttention 3 install: see below). |
| `train_*.log` | Raw Runpod logs for completed ablations (see caveats). |

## Ablations attempted (seed 314, 600s wallclock)

| Run ID | Knobs | Notes |
|--------|--------|--------|
| `ctrl` | Default stack, `GPTQ_CALIB_BATCHES=256`, `BLOCK=128`, `BIGRAM_DIM=112`, `WARMDOWN=4000` | Log ends with **SIGTERM** after `final_int6_roundtrip_exact` — **sliding eval did not complete** in this capture. |
| `a1` | `GPTQ_CALIB_BATCHES=192` | **Has** `final_int6_sliding_window_exact` → use for best **reported** sliding BPB in this bundle. |
| `a2` | `GPTQ_CALIB_BATCHES=320` | Completed through roundtrip; verify log for sliding line when using for analysis. |
| `b1` | `GPTQ_BLOCK_SIZE=256` | Completed through roundtrip in log snapshot. |

**Not included (incomplete or not finished at time of packaging):** `c1`, `c2`, `d1`, `d2` — you can add `train_*.log` files later and update `submission.json`.

## Best reported metric in included logs (for transparency)

From **`train_a1_calib192_seed314.log`**:

- `final_int6_sliding_window_exact` → **val_bpb ≈ 1.56854187** (post-quant sliding window; see log line).

This number is **much worse** than leaderboard SOTA; it reflects **4090 + smaller microbatch**, not a failure of the upstream architecture.

## Reproduce (single GPU, Runpod-style)

```bash
export TRAIN_BATCH_TOKENS=196608
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_PATH=./data/datasets/fineweb10B_sp1024/
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export MAX_WALLCLOCK_SECONDS=600
export BIGRAM_VOCAB_SIZE=3072
export XSA_LAST_N=11

# Example: control config
RUN_ID=ctrl_seed314 SEED=314 WARMDOWN_ITERS=4000 GPTQ_CALIB_BATCHES=256 GPTQ_BLOCK_SIZE=128 BIGRAM_DIM=112 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**FlashAttention 3 (Hopper-focused in upstream README):** on 4090 you may need the image’s installed FA / PyTorch stack; if imports fail, install per the reference record’s `README.md`.

## Why non-record

- Does not meet **8×H100 / 10-minute official** leaderboard bar.
- Incomplete ablation matrix and incomplete **`ctrl` log** (no sliding line).
- Honest documentation of **single-GPU** constraints.

## Next steps (if pursuing a real record later)

1. Re-run **`ctrl`** on 8×H100 without interrupting the process until **`final_int6_sliding_window_exact`** prints.
2. Complete **`c1`–`d2`** or drop them from the story.
3. Run **3 seeds** and collect significance vs prior SOTA per challenge rules.
