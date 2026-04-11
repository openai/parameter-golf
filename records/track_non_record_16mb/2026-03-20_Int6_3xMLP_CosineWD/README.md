# Int6 3xMLP + Cosine Warmdown

val_bpb: **1.1704** (single seed 1337, 8xH100 SXM)

## what i changed

started from the merged SOTA submission (SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit) and added:

- **int6 quantization** - switched from int8 to int6 (range [-31,31]) for both export and training (STE). frees up ~3MB in the artifact budget
- **3x MLP** - used the freed space to widen MLP from 2x to 3x (hidden=1536). model goes from ~18M to ~24M params
- **cosine warmdown** - replaced the linear LR decay with cosine: `0.5 * (1 + cos(pi * progress))`. keeps LR higher early in warmdown, smoother convergence
- **ortho init** - orthogonal weight initialization for all linear layers (except zero-init ones)
- **zstd compression** - zstd level 22 instead of zlib level 9 for the artifact
- **rope base 50k** - passed via env var
- **GQA compat fix** - manual KV head repeat instead of enable_gqa flag (older pytorch compat)

everything else is inherited from the base: 10 layers, muon WD, overtone embed init, sliding window eval stride=64, fp16 embed passthrough, phase-transition resid_mix

## results

| seed | val_loss | val_bpb | steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | 1.9762 | 1.1704 | 7630 | 79.04 |

artifact size: 13.5MB (13,520,480 bytes)
training time: ~10min (603s wallclock cap)
eval time: 61s (sliding window)

only ran one seed so no statistical significance claim. would need 2 more seeds to verify.

## run command

```
EVAL_STRIDE=64 ROPE_BASE=50000 RUN_ID=final_v2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

needs `pip install zstandard` if not already installed.

## notes

- the int6 STE adds overhead per step (~5% slower) but the wider MLP more than makes up for it
- cosine warmdown is pretty standard in LLM pretraining but nobody had tried it here yet
- single seed result so take with a grain of salt
- didn't try seq2048 with this config due to time constraints, might be worth exploring
