# 10L d496 WarmDown3500 + SWA (env-only tuning)

**val_bpb: 1.1590** (1xH100 NVL proxy, single seed) | **15.94 MB** | not yet verified on 8xH100

> **Disclaimer**: This result was obtained on a **single H100 NVL** as a proxy
> for the 8xH100 track. It has not been reproduced on the official 8xH100 setup.
> We expect the 8xH100 result to be somewhat better (more training steps in the
> same wallclock), but we have not verified this. Submitting to share the
> configuration and invite others to reproduce.

## Approach

No code changes. Stock `train_gpt.py` with environment variable overrides only.

The key insight is that `d_model=496` (down from 512) fits under 16MB after
int6+zlib serialization, and `WARMDOWN_ITERS=3500` (up from 3000) gives
slightly better convergence. Disabling TTT and relying on SWA alone keeps
the eval simple and fast.

## 1xH100 Proxy Results

| Metric | Value |
|--------|-------|
| exact_final_val_bpb (int8+zlib roundtrip) | **1.1590** |
| pre_quant_val_bpb | 1.1695 |
| steps | 6,721 / 20,000 |
| step_avg | 758.9ms |
| training wallclock | 5,100s |
| total wallclock (incl eval) | 8,729s |
| bytes_model (int6+zlib) | 15,877,533 |
| bytes_code | 62,791 |
| **bytes_total** | **15,940,324** |

The 1xH100 proxy used `grad_accum=8` to emulate data-parallel batch size,
reaching ~6,700 steps in 5,100s. On 8xH100 the same recipe should reach
~13,000+ steps in 600s, which we expect would improve bpb by 0.01-0.03
based on other submissions' proxy-to-official gaps.

## Reproduction

### 8xH100 (official track, unverified)

```bash
MODEL_DIM=496 NUM_LAYERS=10 WARMDOWN_ITERS=3500 \
SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 \
EVAL_STRIDE=64 TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### 1xH100 proxy (how this result was obtained)

```bash
MODEL_DIM=496 NUM_LAYERS=10 TRAIN_BATCH_TOKENS=524288 \
WARMDOWN_ITERS=3500 MAX_WALLCLOCK_SECONDS=5100 \
SWA_ENABLED=1 SWA_START_FRAC=0.4 SWA_EVERY=50 \
EVAL_STRIDE=64 TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Configuration

| Parameter | Value | Default | Rationale |
|-----------|-------|---------|-----------|
| `MODEL_DIM` | 496 | 512 | Fits under 16MB after int6+zlib |
| `NUM_LAYERS` | 10 | 10 | Same as default |
| `WARMDOWN_ITERS` | 3500 | 3000 | Longer warmdown for better final convergence |
| `SWA_ENABLED` | 1 | 1 | Stochastic weight averaging |
| `SWA_START_FRAC` | 0.4 | 0.4 | Same as default |
| `SWA_EVERY` | 50 | 50 | Same as default |
| `TTT_ENABLED` | 0 | 1 | Disabled to keep eval simple; may benefit from re-enabling on 8xH100 |
| `TRAIN_BATCH_TOKENS` | 524288 | 786432 | Smaller batch = more steps on 1-GPU (proxy only) |

## Limitations

- **Single seed only** — no significance testing yet
- **1-GPU proxy** — 46% of the training steps vs 8xH100
- **No TTT** — enabling TTT with default settings would likely improve by 0.002-0.005 bpb
- **No code innovations** — purely hyperparameter tuning of the stock trainer
