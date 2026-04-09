# RunPod Preflight

This note captures the **pod quality gate** we should run before spending serious credits on a RunPod training session.

The immediate source is [openai/parameter-golf Discussion #743](https://github.com/openai/parameter-golf/discussions/743), posted on **March 25, 2026**, which shares a `30`-second benchmark for screening RunPod pods before training.

## Why we should do this

The discussion's core warning is simple: not all nominally identical H100 pods perform the same. The author reports burning about `$40` on a pod that was roughly `30%` slower than expected before realizing the hardware was underperforming.

Even if the exact percentages vary by host and thermal conditions, the workflow point is strong:

- pod quality is not guaranteed
- the benchmark is cheap
- a bad pod can waste more money than the benchmark will ever cost

So this belongs in our workflow before any real paid run.

## What the benchmark checks

Discussion `#743` says the benchmark reports:

- GPU identity and clock speeds
- GEMM throughput
- memory bandwidth
- disk / network checks
- pod location

The shared one-liner in the discussion is:

```bash
curl -s https://raw.githubusercontent.com/NathanMaine/runpod-gpu-benchmark/main/pod-test.sh | bash
```

Repo link from the discussion:

- <https://github.com/NathanMaine/runpod-gpu-benchmark>

## Practical thresholds

The discussion gives this quick reference for **H100 SXM** pods:

| Metric | Good | Consider switching |
|---|---:|---:|
| GEMM `4096x4096 bf16` | `< 0.50 ms` (`275+ TFLOPS`) | `> 0.70 ms` |
| Memory bandwidth | `> 2800 GB/s` | `< 2000 GB/s` |
| Max GPU clock | about `1980 MHz` | `< 1800 MHz` |

These are not challenge rules. They are operational heuristics for deciding whether the pod is healthy enough to keep.

## Recommended workflow

1. Launch the pod.
2. SSH in and run the benchmark before downloading data or launching training.
3. If the pod lands in the “good” range, keep it.
4. If the pod is clearly below the thresholds, terminate it and re-roll.
5. Only then run our actual training command.

## What not to do

- Do **not** include this benchmark in `train_gpt.py`.
- Do **not** treat it as part of the submission artifact.
- Do **not** add it to the counted record code path.

This is an **ops preflight**, not model code.

## Our current decision rule

For the first real `Spectral Flood Walk` RunPod run:

- keep the pod if it is comfortably inside the “good” band
- re-roll if it is in the “consider switching” band
- if it lands in the middle, prefer re-running once before committing a long experiment

That is intentionally conservative because we are still early in the model-search loop, where bad infrastructure can easily masquerade as bad ideas.
