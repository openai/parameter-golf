# SP8192 + Order-6 Strict Full-Val Byte PPM

**val_bpb = 0.96255** (3-seed mean, std 0.00047) | **15.997 MB mean artifact** | 8xH100 SXM

This submission keeps the SP8192 recurrence / parallel-residual / QK-gain base stack and replaces the prior order-4 PPM setting with a strict full-validation order-6 byte-level PPM mixture at eval time. The PPM state is built online from the already-scored byte prefix, then updated only after each byte is scored.

## Results

| Seed | Post-EMA BPB | PPM BPB | Artifact bytes | Eval time |
| --- | ---: | ---: | ---: | ---: |
| 42 | 1.08754884 | 0.96261595 | 15,996,904 | 474.016s |
| 7 | 1.08763287 | 0.96298648 | 15,999,992 | 464.055s |
| 1337 | 1.08663175 | 0.96205812 | 15,994,492 | 463.261s |
| **Mean** | **1.08727115** | **0.96255352** | **15,997,129** | **467.111s** |
| **Std** | **0.00055533** | **0.00046732** | **2,757** | **5.993s** |

The best seed is 1337 at `0.96205812` BPB. The largest observed total submission size is `15,999,992` bytes, still under the 16,000,000 byte cap.

## Method

The eval path first computes the normal sliding-window neural-network NLLs with stride 64. It then converts the scored token stream into byte contributions and mixes the NN byte probability with an order-6 byte PPM-D probability:

`p_mix = lambda * p_nn + (1 - lambda) * p_ppm`

The gate is binary and prefix-only. With the submitted settings, PPM is trusted more when its longest-context top-symbol confidence is at least `0.9`; otherwise the NN dominates.

| Setting | Value |
| --- | ---: |
| `PPM_ORDER` | `6` |
| `PPM_LAMBDA_HI` | `0.9` |
| `PPM_LAMBDA_LO` | `0.05` |
| `PPM_CONF_THRESHOLD` | `0.9` |
| `PPM_LOG_CACHE_SIZE` | `1048576` |
| `SKIP_QUANTIZED_EVAL` | `1` |
| `SLIDING_BATCH_SEQS` | `32` |

Order 6 was selected after full-val checks. Order 7 and order 8 were slower and worse on seed 42, so they are not part of the submitted result.

## Compliance

- Causal scoring: both NN scoring and PPM scoring use only the prefix available before the current byte.
- Score before update: PPM counts are updated after the byte's mixed log-probability is recorded.
- Single pass: validation bytes are scored once in order; there is no rescoring or best-of-run selection.
- Normalized distribution: PPM-D produces a valid byte distribution and the mixture is performed in probability space.
- Full validation: submitted scores use the full validation stream, not a subset.
- No SLOT, no TTT, no ETLB, and no n-gram cache in the submitted packed artifact.

## Reproduce

```bash
RUN_ID=strict_ppm_order6_seed42 \
SEED=42 \
PPM_ENABLED=1 \
PPM_NATIVE_ENABLED=1 \
PPM_ORDER=6 \
PPM_LAMBDA_HI=0.9 \
PPM_LAMBDA_LO=0.05 \
PPM_CONF_THRESHOLD=0.9 \
PPM_LOG_CACHE_SIZE=1048576 \
SKIP_QUANTIZED_EVAL=1 \
SLIDING_BATCH_SEQS=32 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-27_SP8192_Order6StrictBytePPM/train_gpt.py
```

Change `SEED` and `RUN_ID` to reproduce the other two logs.
