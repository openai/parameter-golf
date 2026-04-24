# SP8192 + Compression-Aware QAT on PR #1493 — 3-seed val_bpb 1.10314

**Author:** yevh ([@yevh](https://github.com/yevh))  
**Base:** PR #1493 by bigbag  
**Track:** non-record 16MB submission for review  
**Date:** 2026-04-22  

## Reviewer TL;DR

This submission evaluates compression-aware QAT on top of the PR #1493 SP8192 stack under the 10-minute / 16MB rules.

The added mechanism is a differentiable entropy surrogate over soft int6 histograms, designed to bias weights toward more compressible post-quantization distributions.

Result across 3 seeds:

- mean quantized TTT `val_bpb`: `1.10314`
- std: `0.00009`
- max artifact size: `15,999,417 B`
- all submitted seeds fit under the 16MB artifact cap
- full train/eval logs are included

The main contribution is testing whether post-quantization compressibility can be optimized directly during training, rather than treated only as a final packing step.

## Results

| Seed | val_bpb | sliding_val_bpb | artifact bytes | train steps |
|---:|---:|---:|---:|---:|
| 1338 | 1.10325 | 1.10482 | 15,996,195 | 2,675 |
| 1339 | 1.10312 | 1.10455 | 15,999,417 | 2,665 |
| 1340 | 1.10307 | 1.10450 | 15,994,396 | 2,671 |
| Mean | 1.10314 | 1.10462 | 15,996,669 | 2,670 |

## What changed vs PR #1493

The PR #1493 stack is preserved. The intended algorithmic change is the compression-aware QAT term:

- soft assignment of weights to an int6 grid
- entropy penalty over the resulting histogram
- activation after warmup
- applied to large 2D linear matrices only

The QAT configuration used in this submission:

- `lambda = 0.001`
- `beta = 10.0`
- `warmup = 200`
- objective: mean Shannon entropy of soft-int6 histograms

## Research context and pivot

This run came from a broader experiment campaign around compression-aware training for Parameter Golf.

The original direction was to explore whether artifact-size pressure could be improved not only through final packing, but also during training. The campaign tested several related ideas:

1. **Compression-aware QAT**  
   This became the final submitted branch. It adds a differentiable entropy surrogate over soft int6 histograms and applies it during training after warmup.

2. **LoRA / TTT interaction investigation**  
   During experiments, I found an inference-mode interaction around TTT-style adaptation. This is documented as a supporting research note, not as a separate scored run.

3. **3DCF compression-vs-packing investigation**  
   I also tested whether ideas from semantic/document compression could transfer to Parameter Golf artifact compression. The conclusion was that semantic compression and binary artifact packing optimize different things; this is included as a research note.

The final pivot was to submit the most reproducible and scoreable branch: **CompQAT on top of the PR #1493 SP8192 stack**.

The key finding is that the compression surrogate is stable and directionally meaningful, but the current implementation adds enough per-step overhead that it reduces the number of training steps completed within the 600s budget. This likely explains why the final BPB does not improve over the base PR #1493 stack.

## What did not work yet

The compression objective behaved as intended, but the current implementation was too expensive inside the 600s training budget.

In practice, the surrogate reduced effective training throughput. So even though it applied useful compression pressure, the model completed fewer optimization steps than the base PR #1493-style run. The result is a stable but non-improving BPB.

The next likely improvement is to make the surrogate cheaper:

- compute it every N steps instead of every step
- apply it only to the largest MLP matrices
- use a cheaper histogram approximation
- sweep smaller lambda values

## Longer write-up

A longer write-up of the full experiment campaign is included here:

- [`article.md`](./article.md)

Supporting research notes are included under:

- [`research/2026-04-22_compqat_campaign/`](../../../research/2026-04-22_compqat_campaign/)

## Files

- `README.md` — this summary
- `submission.json` — machine-readable submission metadata
- `train_gpt.py` — runnable training script
- `train.log` — canonical seed 1340 run
- `train_seed1338.log` — full seed 1338 log
- `train_seed1339.log` — full seed 1339 log
- `train_seed1340.log` — full seed 1340 log
- `compression_surrogate.py` — readable standalone version of the surrogate for review; the actual training implementation is embedded in `train_gpt.py`
- `article.md` — longer write-up of the experiment campaign

## Reproduction

Example command:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Expected environment:

```text
8×H100 80GB SXM
PyTorch 2.9.1+cu128
CUDA 12.8
```
