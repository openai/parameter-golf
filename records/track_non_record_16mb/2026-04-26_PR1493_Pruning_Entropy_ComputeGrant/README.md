# Draft non-record checkpoint: PR1493 pruning + q-symbol entropy coding

**Status:** draft / non-record checkpoint for compute-grant request  
**Track:** non-record 16MB / unlimited-compute checkpoint  
**Author:** VedantKmr0  
**Current reference BPB:** `1.08502253` after quantization + sliding-window eval + TTT on the 8xH100 PR1493-style run

This folder is a draft checkpoint, not a final leaderboard submission. It documents the current pruning and compression directions and includes the scripts needed to regenerate the PR-facing experiments from an existing PR1493-style capture artifact.

The goal is to request additional compute to validate the best pruning/compression combinations, run recovery/fine-tuning experiments, and then produce a final reproducible submission.

## Summary

The current direction is to combine:

1. **Structured MLP hidden-channel pruning** with per-block or soft per-block caps.
2. **Integer entropy coding / rANS-style coding** for quantized `q` tensors.

The main empirical finding so far is that naive global MLP channel pruning is too concentrated. It mostly collapses into a single late block, especially `blocks.10.mlp`, and causes much larger BPB degradation. Adding per-block caps spreads pruning across layers and is substantially better.

The public soft-cap variant blends within-block and global channel ranks, then applies a relaxed per-block cap. The pushed 5% soft-cap result used:

- `score_weights = activation_weighted_score=0.70,norm_score=0.30`
- `local_rank_weight = 0.75`
- `cap_multiplier = 1.75`
- `floor_multiplier = 0.0`

## Baseline / training run

Final metrics from the 8xH100 PR1493-style run:

| Stage | Val loss | Val BPB | Eval time |
|---|---:|---:|---:|
| pre-quantization post-EMA | `2.82175375` | `1.09238881` | `6.776s` |
| quantized | `2.84964269` | `1.10318549` | `25.077s` |
| quantized + sliding window | `2.80624420` | `1.08638458` | `128.040s` |
| quantized + TTT | `2.80272589` | `1.08502253` | `339.862s` |

Size and quantization notes:

- raw serialized model: `135,431,033` bytes
- code size: `59,270` bytes
- quantized raw state: `36,142,183` bytes
- quantized + Brotli model: `15,975,480` bytes
- total submission size quantized + Brotli: `16,034,750` bytes
- GPTQ int6: attention Q/K/V/O and MLP fc/proj weights
- GPTQ int8: token embedding
- float16 passthrough: q gains, residual/MLP/attention scales, residual mixing, skip weights/gates

## Pruning experiments

### Method

Each MLP hidden channel corresponds to:

- one row in `mlp.fc.weight`
- one column in `mlp.proj.weight`

For a proposed structured pruning mask, I evaluate quality by zero-ablation:

- zero selected `mlp.fc.weight` rows
- zero matching `mlp.proj.weight` columns
- run validation BPB on the final EMA checkpoint

This keeps the pruning relatively good and fairly fast. The final implementation would remove these channels structurally from the serialized model.

### Naive global pruning

The first experiment globally ranked all MLP channels by importance score and pruned the lowest-scoring channels.

This was not robust: pruning concentrated mostly in `blocks.10.mlp`, causing large degradation.

Uncapped global pruning results:

| Prune fraction | Delta BPB |
|---:|---:|
| 1% | `+0.007035` |
| 2% | `+0.015820` |
| 5% | `+0.057837` |
| 8% | `+0.135241` |

### Per-block capped pruning

I then added a per-block cap equal to the requested global prune fraction. This forces pruning to be spread across MLP blocks instead of collapsing into one block.

| Setting | BPB | Delta BPB | Estimated int6 packed bytes saved |
|---|---:|---:|---:|
| baseline | `1.09238693` | — | — |
| 1% capped | `1.09673307` | `+0.00434614` | `~172.8 KB` |
| 2% capped | `1.10103679` | `+0.00864986` | `~346.4 KB` |
| 5% capped | `1.11593685` | `+0.02354992` | `~864.8 KB` |
| 8% capped | `1.13440505` | `+0.04201812` | `~1.38 MB` |

This is clearly better than uncapped pruning at every tested fraction.

Current interpretation:

- **1–2% capped pruning** looks like the safest current tradeoff.
- **5% capped pruning** may be recoverable with short fine-tuning or recovery training.
- **8% capped pruning** is probably too aggressive without recovery.

### Soft-cap direction

The soft-cap script generalizes hard per-block caps into a tunable allocation policy.

Instead of forcing every block to prune exactly the same fraction, it combines:

- within-block channel ranking,
- global channel ranking,
- a relaxed per-block cap,
- an optional per-block floor.

The pushed 5% `soft_cap_global` full-validation result used the parameters listed above and produced a single-GPU full-validation result that seems promising, but it is not directly comparable to the earlier world-size-8 pruning baseline until both are evaluated under the same validation settings.

## Compression experiments

I also tested entropy coding over the integer quantized weights.

The current quantized artifact is Brotli-compressed. I compared this against static arithmetic/rANS-style coding estimates over signed quantized `q` symbols.

Current estimates:

- current quantized + Brotli artifact: `15,975,480` bytes
- static arithmetic/rANS q-symbol model, per tensor: `15,681,130` bytes
- static arithmetic/rANS q-symbol model, per class: `15,674,853` bytes
- raw q bytes: `35,913,728` bytes

The current conclusion is that integer entropy coding is a viable complement to pruning and might be at least on par with brotli with further experimentation required. The next step is to implement a fast/decode-safe rANS container and validate final artifact size and correctness.

## Why I need additional compute

The existing experiments identify promising directions but require more validation:

1. Run and test further pruning algorithms to improve the soft-cap results.
2. Evaluate whether 3–5% capped/soft-capped pruning can be recovered with short fine-tuning.
3. Combine the safest pruning point with integer entropy coding.
4. Validate final artifact size, BPB, and reproducibility on the official evaluation path.
5. Test architectures to exploit the gained storage space.

## Included files

- `README.md` — this write-up.
- `submission.json` — non-record metadata for the draft checkpoint.
- `train_gpt.py` — public entrypoint; runs the local PR1493 stack by default and exposes helper subcommands.
- `train_gpt_pr1493.py` — local PR1493 training/evaluation stack used by this checkpoint.
- `grant_generate_mlp_cap_masks.py` — capped MLP mask generator.
- `grant_generate_mlp_softcap_masks.py` — soft-cap MLP mask generator.
- `grant_eval_mlp_zero_ablation.py` — zero-ablation evaluator for pruning masks.
- `grant_estimate_q_entropy.py` — q-symbol static arithmetic/rANS codelength estimator.
- `train_pr1493_capture.log` — short pointer log with key capture/run status.

## Reproduction commands

Normal training entrypoint:

```bash
python records/track_non_record_16mb/2026-04-26_PR1493_Pruning_Entropy_ComputeGrant/train_gpt.py
```

These helper commands assume the capture artifact exists at:

```bash
artifacts/vast_8xh100_pr1493_capture_20260425_r3
```

Generate per-block capped masks:

```bash
python records/track_non_record_16mb/2026-04-26_PR1493_Pruning_Entropy_ComputeGrant/train_gpt.py cap-masks \
  --artifact-dir artifacts/vast_8xh100_pr1493_capture_20260425_r3 \
  --fractions 0.01,0.02,0.05,0.08
```

Generate soft-cap masks:

```bash
python records/track_non_record_16mb/2026-04-26_PR1493_Pruning_Entropy_ComputeGrant/train_gpt.py softcap-masks \
  --artifact-dir artifacts/vast_8xh100_pr1493_capture_20260425_r3 \
  --fractions 0.005,0.01,0.015,0.02,0.03,0.05 \
  --score-weights activation_weighted_score=0.70,norm_score=0.30 \
  --local-rank-weight 0.75 \
  --cap-multiplier 1.75 \
  --floor-multiplier 0.0
```

Evaluate pruning masks:

```bash
torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-04-26_PR1493_Pruning_Entropy_ComputeGrant/train_gpt.py eval-pruning \
  --artifact-dir artifacts/vast_8xh100_pr1493_capture_20260425_r3
```

Estimate q-symbol entropy coding:

```bash
python records/track_non_record_16mb/2026-04-26_PR1493_Pruning_Entropy_ComputeGrant/train_gpt.py q-entropy \
  --quantized artifacts/vast_8xh100_pr1493_capture_20260425_r3/final_model.int6.ptz
```

## Status

Draft / non-record. Additional compute is needed before this can become a final submission.
