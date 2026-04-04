# Shared-Block Recurrent Transformer, Int6+zstd (8xH100)

**final_int6_roundtrip_exact val_bpb = 1.3693** | **3.59 MB** artifact | **8xH100**, 600s wallclock

This folder captures an 8xH100 RunPod run of the current `train_gpt.py`.

## Summary

The model uses a shared universal transformer block reused across multiple passes instead of using a different block at each layer. The main ingredients in this run are:

1. **Shared universal block reused across 8 passes**. Depth is created by repeatedly applying one block with pass-dependent conditioning rather than stacking many unique blocks.
2. **Pass-dependent rotations and depth embeddings**. Each reuse of the shared block sees a slightly different representation.
3. **Partial RoPE** with `ROPE_DIMS=16`.
4. **LN scale** and **depth modulation**.
5. **BigramHash + SmearGate** features.
6. **EMA/SWA averaging**.
7. **Int6 quantization + zstd compression**.

## Configuration

```bash
RUN_ID=hbt_seed1337 SEED=1337 LOG_FILE=logs/train_seed1337.log \
ITERATIONS=50000 MAX_WALLCLOCK_SECONDS=600 WARMUP_STEPS=0 AUTO_LIGHT_LOCAL=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Logged model settings from the run:

- `model_params=5866250`
- `num_passes=8`
- `mlp_mult=8`
- `rope_dims=16`
- `bigram_vocab_size=2048`
- `eval_stride=64`
- `compressor=zstd`

## Results

Timed training stopped at:

- `7601/50000` due to the 600s wallclock cap

Key metrics from `train.log`:

| Metric | Value |
|---|---:|
| Pre-quant val_loss at stop | 2.2873 |
| Pre-quant val_bpb at stop | 1.3547 |
| **final_int6_roundtrip_exact val_loss** | **2.31193436** |
| **final_int6_roundtrip_exact val_bpb** | **1.36925776** |
| Model params | 5,866,250 |
| Peak memory | 16,639 MiB |
| Int6+zstd model bytes | 3,531,468 |
| Code bytes | 59,835 |
| **Total artifact bytes** | **3,591,303** |
| Under 16MB | YES |

## Notes

- This submission reports the exact printed `final_int6_roundtrip_exact` metric from the attached log.
- The process printed the final primary metric and artifact-size lines successfully.
- In this recorded run, the shell session was interrupted before sliding-window evaluation and final cleanup completed, so no sliding-window score is claimed here.

## Included Files

- `train_gpt.py` - exact training / quantization script used for the run
- `train.log` - training log for the submitted run
- `submission.json` - metadata for this submission
