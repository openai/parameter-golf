# Sequence-level V-shaped Curriculum Learning + LeakyReLU²

## Score: val_bpb = 1.3557 (1×H100, single seed)

Trained on 1×H100 80GB in 600 seconds. 15.25MB artifact (int6+zstd). Run on 1×H100 due to compute constraints. Built upon [PR #623](https://github.com/openai/parameter-golf/pull/623).

## Motivation

Standard training feeds random batches regardless of training phase. In a 600-second window (~1100 steps), the model encounters different optimization regimes — high LR early, decay mid-training, SWA late — but the data difficulty remains uniform throughout. Human learners don't study random material; they start with fundamentals and progress to harder content. Can we apply the same principle to LLM training?

**meaningful difficulty variance exists at the sequence level within each batch** — individual sequences range from repetitive boilerplate (low entropy) to diverse technical text (high entropy). Operating at this granularity lets us complete multiple full curriculum cycles within our data budget.

## Approach

Implements online sequence-level curriculum learning that operates within each batch.

### Why 2× oversampling?

To select sequences by difficulty, we need a pool larger than the batch size. Loading exactly 256 sequences gives no room to filter — every sequence must be used. By loading 512 (2×), we can rank all sequences by entropy and pick the 256 that best match the current difficulty target. The 2× factor is the minimum that allows meaningful selection: with k× oversampling, each selected sequence is drawn from the top 1/k of the difficulty distribution around the target percentile. Higher k (3×, 4×) would give tighter selection but waste proportionally more data and I/O bandwidth. 2× balances selection quality against data efficiency — we discard 50% of loaded tokens but each retained sequence is more informative for the current training phase.

### Mathematical Formulation

**Difficulty metric — per-sequence unigram entropy**: For each sequence s of length L=2048:

```
H(s) = -Σ_{t=1}^{V} p_s(t) · log₂(p_s(t))
```

where p_s(t) = count(token t in s) / L, V=1024. High entropy = diverse vocabulary = harder to predict. Unlike shard-level entropy (which averages over 100M tokens per shard, washing out variance), sequence-level entropy has meaningful variance — individual sequences range from highly repetitive (low H) to diverse/technical (high H).

**V-shaped difficulty target**: A continuous function d(step) ∈ [0,1] mapping training progress to target difficulty percentile:

```
d(step) = step / (p · T)                       if step ≤ p · T
d(step) = 1 - (step/T - p) / (1 - p)           otherwise
```

where T=1100 (estimated total steps), p=0.45 (peak fraction). This gives:
- Steps 0→495: ramp 0→1 (easy→hard). LR is high; easy data prevents gradient explosion while establishing basic representations.
- Steps 495→1100: ramp 1→0 (hard→easy). As LR decays and SWA begins (~step 550), returning to easier data produces more coherent checkpoint averages.

**Selection mechanism**: At each training step:
1. Load 2× the needed sequences from the data stream (512 instead of 256 for diversity)
2. Compute H(s) for all 512 sequences 
3. Sort by entropy, select 256 sequences centered around percentile d(step)


The selection window slides smoothly through the difficulty distribution as training progresses, completing multiple full V-cycles even within 6 shards of data.

### Stability alignment

The V-shape is designed to align with three training phases:
- **High LR phase** (steps 0–495): d ramps up. Gradient variance Var(∇L) scales with sequence difficulty. Starting easy ensures stable optimization at high learning rates.
- **LR decay + SWA onset** (steps 495–550): d peaks then decreases. The transition from hard→easy data coincides with the shift to weight averaging.
- **SWA region** (steps 550+): d decreases to 0. Consecutive SWA checkpoints train on similar-difficulty data, producing a more coherent average.

## Tradeoff

Consumes 2× data per step but only trains on half — 50% data efficiency loss (we see ~575M tokens but only train on ~288M). The hypothesis is that each selected sequence is more informative for the current training phase than a random one.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_layers | 11 (10 unique) |
| model_dim | 512 |
| mlp_activation | LeakyReLU(0.5)² |
| seq_curriculum_oversample | 2.0 |
| est_total_steps | 1100 |
| peak_frac | 0.45 |
| train_batch_tokens | 524,288 |
| matrix_lr / scalar_lr | 0.025 |
| swa_every | 50, start_frac=0.2 |

## Key Metrics

- **val_bpb: 1.3557** (post int6+zstd roundtrip)
- Pre-quant val_bpb: 1.3280
- Quantization penalty: 0.0277 bpb
- Training: 1,021 steps in 600s (588 ms/step)
- Artifact size: 15,250,895 bytes (15.25MB)
- SWA: averaged 12 checkpoints
- Peak memory: 14,656 MiB

## Observation

Worse than baseline (1.3345). Two likely causes:

1. **Overhead reduces steps**: The 2× oversampling + CPU entropy computation + sorting adds ~50ms/step (588ms vs 540ms baseline), costing ~80 training steps over 600s (1021 vs 1100). The curriculum signal doesn't compensate for the lost steps.


**Implication**: For curriculum to work at this scale, it must be zero-overhead — perhaps by precomputing difficulty scores offline and reordering the data stream rather than filtering at runtime.
