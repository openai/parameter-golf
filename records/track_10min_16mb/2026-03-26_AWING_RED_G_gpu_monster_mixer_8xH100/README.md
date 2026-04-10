# A-WING RED_G: GPU Monster Mixer

**val_bpb: 0.7614** (seed 1337, `final_int6_sliding_window_ngram9_exact`) | **15.18 MB** | 8xH100 SXM

## Results

| Seed | final val_bpb | Int6 sliding bpb | Int6 roundtrip bpb | Steps | Train Time | N-gram Eval Time | Artifact |
|------|--------------:|-----------------:|-------------------:|------:|-----------:|-----------------:|---------:|
| 1337 | 0.76141536 | 1.13088592 | 1.15457064 | 5325 | 570.065s | 211.727s | 15,180,405 B |

## Mixer Performance (Goal: fast startup)

| Metric | Value |
|-------|------:|
| Prefill mode | `sharded+allreduce-gpu` |
| Buckets | 2,097,152 |
| Orders | 2..9 |
| Max shards | 80 |
| Tokens / shard cap | 50,000,000 |
| Prefilled tokens | 4,000,000,000 |
| Prefill time | 5.8s |
| Prefill sync | 1.0s |
| Effective aggregate prefill throughput | ~689.7M tok/s |

## Key Takeaways

- The GPU mixer startup bottleneck is resolved: prefill + sync is **6.8s total**, well under the 90s cap.
- N-gram stack gives a large gain: `1.13088592 -> 0.76141536` (delta `-0.36947056`, **-32.67%**).
- Training remained within budget and stopped by wallclock as intended at 570s.
- Memory and size constraints passed:
  - Peak allocated: 21,141 MiB
  - Submission size (int6+zstd): 15,180,405 bytes

## Run Configuration Snapshot

- Script: `experiments/A_wing/RED_G/run.sh`
- Seed: `1337`
- GPUs: `8xH100`
- Mixer: `Linear(512->9)`, orders `2..9`
- `MIXER_GPU_MODE=1`
- `MIXER_PREFILL_MAX_SHARDS=80`
- `MIXER_PREFILL_MAX_SECONDS=90`
- `MIXER_PREFILL_MIN_SHARDS=4`
- `MIXER_PREFILL_TOKENS_PER_SHARD=50000000`
- `MIXER_BUCKETS=2097152`
- `NGRAM_EVAL_BUCKETS=16777216`
- `MAX_WALLCLOCK_SECONDS=570`

## Raw Metrics Captured

- `final_int6_roundtrip_exact val_loss:1.94944417 val_bpb:1.15457064`
- `final_int6_sliding_window_exact val_loss:1.90944845 val_bpb:1.13088592`
- `final_int6_sliding_window_ngram9_exact val_loss:1.28561453 val_bpb:0.76141536`
- `stopping_early: wallclock_cap train_time:570065ms step:5325/20000`

## Reproduce

```bash
bash experiments/A_wing/RED_G/run.sh
```
