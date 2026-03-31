# Non-Record Submission: Hyperbolic Q/K Lite (1xH100, 10min)

## Summary

This folder documents a lightweight hyperbolic-attention exploration built on the standard GPT-style autoregressive training loop.

The main idea is intentionally narrow: apply a Lorentz-style hyperbolic mix only to the attention `q` and `k` projections, while keeping the rest of the model mostly Euclidean for stability and speed.

Main run:

- `hyp_qk_lite_10min_wd3000.txt`
- **final_int8_zlib_roundtrip val_bpb: 1.32877977**
- **Artifact size: 11,673,884 bytes**
- **1xH100 80GB**, **600.217s**, **1378 steps**

This is a non-record submission. It does not target the current SOTA; it is meant as a compact research package showing a concrete hyperbolic geometry insertion that trains end-to-end and improves substantially over early smoke runs.

## Approach

The implementation keeps the baseline structure recognizable:

- 9-layer transformer
- 512 model dim
- 8 attention heads / 4 KV heads (GQA)
- tied embeddings
- standard tokenizer-aware `val_bpb` evaluation

The hyperbolic change is small and local:

- introduce trainable `hyperbolic_qk_mix`
- introduce trainable `hyperbolic_radius`
- transform only attention `q` and `k`
- leave values, MLP, residual path, and decoder in their simpler Euclidean form

The motivation was to test whether a small amount of geometric distortion in the attention similarity path could help without paying the large optimization and systems cost of a full hyperbolic stack.

## Results

### Main run

From `logs/hyp_qk_lite_10min_wd3000.txt`:

- step stop: `1378`
- wallclock: `600.217s`
- pre-quant final validation: `val_loss=2.2377`, `val_bpb=1.3253`
- roundtrip validation: `val_loss=2.24358899`, `val_bpb=1.32877977`
- total code + compressed model bytes: `11,673,884`

### Short-run ablations

| Log | Setting | Wallclock | Final roundtrip val_bpb |
|-----|---------|-----------|--------------------------|
| `hyp_qk_lite_smoke_30s.txt` | 30s smoke | 30.362s | 3.17082135 |
| `hyp_qk_180src_smoke_30s.txt` | 30s smoke from 180s-source variant | 30.385s | 3.21542997 |
| `hyp_qk_lite_repro_180s.txt` | 180s reproduction | 180.058s | 1.59374340 |
| `hyp_qk_lite_mix015_r008_180s.txt` | stronger mix / smaller radius | 180.021s | 1.65123372 |
| `hyp_qk_lite_10min_wd3000.txt` | 600s main run with long warmdown | 600.217s | 1.32877977 |

The strongest result in this folder came from the long 600s run with warmdown. More aggressive hyperbolic strength was generally worse than the lighter `q/k`-only setting.

## What Was Tried

This folder is the distilled "q/k-only" branch of a broader March 28 hyperbolic exploration. The broader local exploration also included:

- local hyperbolic Q/K baselines
- 180s hyperbolic q-only and q/k-only sweeps
- merge attempts with other code stacks
- hybrid local-linear attention variants
- Hypformer-like full-hyperbolic long-run experiments
- HypGPT-style full-stack q/k/v + hyperbolic MLP experiments

Those wider branches were useful for understanding failure modes, but the cleanest stable result was this lightweight `q/k`-only package.

## Files

- `train_gpt.py`: main lightweight hyperbolic q/k training script
- `train_gpt_from_180s.py`: shorter follow-up variant
- `train_gpt_mixschedule.py`: scheduled hyperbolic-mix experiment
- `logs/`: smoke, 180s, and 600s runs used for comparison

## Reproduction

Main run:

```bash
cd records/track_non_record_16mb/2026-03-28_hyperbolic_qk_lite_10min_1gpu
RUN_ID=hyp_qk_lite_10min_wd3000 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_ITERS=3000 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=524288 \
HYPERBOLIC_QK_MIX=0.02 \
HYPERBOLIC_RADIUS_INIT=0.10 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Notes

- This is a 1xH100 research run, not an 8xH100 leaderboard attempt.
- The folder is intentionally self-contained and keeps the hyperbolic intervention small.
- The main value of this submission is showing a concrete, runnable hyperbolic-attention modification with multiple ablations and logs, rather than claiming state-of-the-art performance.
