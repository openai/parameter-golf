# Parameter Golf - Initial Training Report

## Run Summary

| Field | Value |
|-------|-------|
| **try_id** | `baseline_sp1024_try1` |
| **Date** | 2026-03-20 23:24–23:52 UTC |
| **GPU** | 1x NVIDIA H100 80GB SXM |
| **Pod ID** | `1na4h7m0t0vauz` |
| **Pod Cost** | $2.69/hr (~$1.26 for this run) |
| **Training Wall Time** | 600.3s (10 min 0.3s, hit wallclock cap) |
| **Steps Completed** | 1404 / 20000 |
| **Seed** | 1337 |

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | GPT (9 blocks, width 512) |
| Attention | GQA: 8 heads, 4 KV heads |
| MLP Expansion | 2x |
| Vocab Size | 1024 (SentencePiece BPE) |
| Sequence Length | 1024 |
| Total Parameters | 17,059,912 (~17M) |
| Tied Embeddings | Yes |
| Batch Tokens | 524,288 |
| Grad Accum Steps | 8 |
| Peak GPU Memory | 10,239 MiB allocated / 10,730 MiB reserved |

## Training Metrics

| Step | train_loss | val_loss | val_bpb | train_time |
|------|-----------|----------|---------|------------|
| 0 | — | 6.9357 | 4.1077 | 0ms |
| 10 | 5.9957 | — | — | 4.5s |
| 200 | 2.7898 | — | — | 83.3s |
| 400 | 2.3871 | — | — | 168.9s |
| 600 | 2.4803 | — | — | 254.4s |
| 800 | 2.3204 | — | — | 340.6s |
| 1000 | 2.3338 | 2.3023 | 1.3636 | 426.2s |
| 1200 | 2.2551 | — | — | 512.8s |
| 1400 | 2.2873 | — | — | 598.5s |
| **1404** | — | **2.2392** | **1.3262** | **600.3s** |

## Final Results (int8+zlib roundtrip)

| Metric | Value |
|--------|-------|
| **val_loss** | 2.24120646 |
| **val_bpb** | 1.32736871 |
| Eval Time | 11,054ms |
| Raw Model Size | 67,224,983 bytes (~64 MB) |
| int8+zlib Compressed | 13,728,606 bytes (~13.1 MB) |
| Code Size | 47,686 bytes (~47 KB) |
| **Total Submission Size** | **13,776,292 bytes (~13.1 MB)** |
| Compression Ratio | 3.91x |
| Under 16MB Limit? | Yes |

## Observations

1. **Wallclock limited**: Training hit the 600s cap at step 1404/20000. Only ~7% of planned iterations completed. With 8 GPUs (the competition target), throughput would increase significantly.

2. **Loss still decreasing**: val_bpb improved from 1.3636 (step 1000) to 1.3262 (step 1404), suggesting more training time would further improve results.

3. **Well under size limit**: The compressed submission is 13.8 MB vs the 16 MB cap, leaving ~2.2 MB of headroom for larger models or additional code.

4. **Baseline vs leaderboard**: This baseline 1.3274 BPB is far from the current SOTA (~1.1428 BPB). Top entries use int5/int6 quantization-aware training, 3x MLP expansions, and multi-GPU scaling.

## Step Throughput

- Average step time: ~427.5ms
- Tokens per second: ~1.23M tokens/step / 0.4275s ≈ 1.23M tok/s (single GPU)

## Stop Reason

Training stopped due to `wallclock_cap` (600s limit). Pod was stopped after training to save cost.
